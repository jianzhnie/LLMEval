## 数据并行部署

vLLM 支持**数据并行**部署，将模型权重复制到不同的实例/GPU 上，以处理独立的请求批次。

该模式适用于**密集模型**和 **MoE (Mixture of Experts) 模型**。

对于 MoE 模型，特别是像 DeepSeek 这样采用 **MLA (多头潜在注意力)** 的模型，将注意力层用于数据并行（DP），而将专家层用于**专家并行（EP）**或**张量并行（TP）**可能更具优势。在这种情况下，数据并行节点并非完全独立。即使需要处理的请求少于 DP 节点数，正向传播也必须对齐，并且所有节点上的专家层在每次正向传播时都必须同步。

默认情况下，专家层会形成一个 **(DP x TP)** 大小的张量并行组。要启用专家并行，需在所有节点上都包含 `--enable-expert-parallel` CLI 参数。

在 vLLM 中，每个 DP 节点被部署为一个独立的**“核心引擎”**进程，通过 **ZMQ socket** 与前端进程通信。数据并行注意力可以与张量并行注意力结合使用，此时每个 DP 引擎拥有的每个 GPU 工作进程数等于配置的 TP 大小。

对于 MoE 模型，当任何节点上有请求正在进行时，我们必须确保在所有当前没有计划请求的节点上执行空的**“虚拟”正向传播**。这是由一个独立的 **DP 协调器**进程处理的，该进程与所有节点通信，并通过每 N 步执行一次的集体操作来确定所有节点何时都处于空闲状态并可以暂停。当 TP 与 DP 结合使用时，专家层会形成一个大小为 **(DP x TP)** 的 EP 或 TP 组。

在所有情况下，在 DP 节点之间进行**请求负载均衡**都是有益的。对于在线部署，可以通过考虑每个 DP 引擎的状态（特别是其当前计划和等待（排队）的请求以及 KV 缓存状态）来优化这种均衡。每个 DP 引擎都有一个独立的 **KV 缓存**，通过智能地定向提示，可以最大限度地利用前缀缓存的优势。

本文档重点介绍在线部署（使用 API 服务器）。DP + EP 也支持离线使用（通过 LLM 类），示例可参见 `examples/offline_inference/data_parallel.py`。

在线部署支持两种不同的模式：**自带内部负载均衡**和**外部按节点进程部署及负载均衡**。

### 内部负载均衡

vLLM 支持**“自包含”**的数据并行部署，它暴露一个**单一的 API 端点**。

只需在 `vllm serve` 命令行参数中包含 `--data-parallel-size=4` 即可进行配置。这将需要 4 个 GPU。它可以与张量并行结合使用，例如 `--data-parallel-size=4 --tensor-parallel-size=2`，这将需要 8 个 GPU。

在多节点上运行单个数据并行部署需要在每个节点上运行不同的 `vllm serve`，并指定该节点应运行哪些 DP 节点。在这种情况下，仍将有一个**单一的 HTTP 入口点**——API 服务器将只在一个节点上运行，但它不一定需要与 DP 节点位于同一位置。

这将在单个 8-GPU 节点上运行 DP=4，TP=2：

```bash
vllm serve $MODEL --data-parallel-size 4 --tensor-parallel-size 2
```

这将在 DP 节点 0 和 1 位于主节点，而节点 2 和 3 位于第二个节点上时运行 DP=4：

```bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                   --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 2 \
                   --data-parallel-start-rank 2 \
                   --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

这将在第一个节点上只运行 API 服务器，所有引擎都在第二个节点上运行 DP=4：

```Bash
# Node 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 0 \
                   --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Node 1
vllm serve $MODEL --headless --data-parallel-size 4 --data-parallel-size-local 4 \
                   --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

这种 DP 模式也可以通过指定 `--data-parallel-backend=ray` 与 Ray 一起使用：

```bash
vllm serve $MODEL --data-parallel-size 4 --data-parallel-size-local 2 \
                   --data-parallel-backend=ray
```

使用 Ray 时有几个显著的区别：

- 只需一个启动命令（在任何节点上）即可启动所有本地和远程 DP 节点，因此比在每个节点上启动更方便。
- 无需指定 `--data-parallel-address`，运行该命令的节点将被用作 `--data-parallel-address`。
- 无需指定 `--data-parallel-rpc-port`。
- 远程 DP 节点将根据 Ray 集群的节点资源进行分配。

目前，内部 DP 负载均衡是在 API 服务器进程内完成的，并基于每个引擎中的运行队列和等待队列。未来可以通过整合**KV 缓存感知逻辑**使其更加复杂。

当使用此方法部署大型 DP 规模时，API 服务器进程可能成为瓶颈。在这种情况下，可以使用正交的 `--api-server-count` 命令行选项进行扩展（例如 `--api-server-count=4`）。这对于用户是透明的——仍然暴露一个**单一的 HTTP 端点/端口**。请注意，这种 API 服务器扩展是“内部的”，并且仍然**局限于“主”节点**。

### 外部负载均衡

对于更大规模的部署，由外部处理数据并行节点的编排和负载均衡可能更有意义。

在这种情况下，将每个 DP 节点视为一个独立的 vLLM 部署，拥有自己的端点，并让**外部路由器**在它们之间进行 HTTP 请求的均衡，利用每个服务器的适当实时遥测数据进行路由决策，会更方便。

对于非 MoE 模型，这已经可以轻松完成，因为每个部署的服务器都是完全独立的。为此，无需使用任何数据并行 CLI 选项。

我们支持 MoE DP+EP 的同等拓扑，可以通过以下 CLI 参数进行配置。

如果 DP 节点位于同一位置（同一节点/IP 地址），则使用默认的 RPC 端口，但必须为每个节点指定不同的 HTTP 服务器端口：

```Bash
# Rank 0
CUDA_VISIBLE_DEVICES=0 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                         --port 8000
# Rank 1
CUDA_VISIBLE_DEVICES=1 vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                         --port 8001
```

对于多节点情况，还必须指定节点 0 的地址/端口：

```Bash
# Rank 0  (with ip address 10.99.48.128)
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 0 \
                   --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
# Rank 1
vllm serve $MODEL --data-parallel-size 2 --data-parallel-rank 1 \
                   --data-parallel-address 10.99.48.128 --data-parallel-rpc-port 13345
```

协调器进程也在此场景中运行，与 DP 节点 0 引擎位于同一位置。

在上图中，每个虚线框对应于一个独立的 `vllm serve` 启动——例如，这些可以是独立的 **Kubernetes Pod**.
