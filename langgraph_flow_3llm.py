# complex_dialog_flow_perf.py

import asyncio
import csv
import json
import os
import platform
import uuid
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Dict, List, NotRequired, TypedDict

# 使用try-except避免Windows系統錯誤
try:
    import uvloop

    has_uvloop = True
except ImportError:
    has_uvloop = False

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig, RunnableLambda
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger
from pydantic import BaseModel, Field

# --- 配置與初始化 ---

# 配置日誌
logger.add(
    "complex_dialog_flow.log", rotation="10 MB", retention="7 days", level="DEBUG"
)

# 加載環境變數
load_dotenv()

# 檢查並設置 OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY_QA")
if not api_key:
    logger.warning("OPENAI_API_KEY_QA 環境變數未設置，嘗試使用 OPENAI_API_KEY")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("無法找到有效的 OpenAI API 金鑰")
        raise ValueError("No valid OpenAI API key found")

# 明確設置 OPENAI_API_KEY 環境變數
os.environ["OPENAI_API_KEY"] = api_key
logger.info("成功設置 OpenAI API 金鑰")

# 建立輸出目錄
output_dir = Path("dialog_outputs")
output_dir.mkdir(exist_ok=True)

# 性能計時數據儲存
perf_data = []

# --- 定義對話流程狀態 ---


class Message(BaseModel):
    """對話消息模型"""

    role: str  # 'user', 'agent1', 'agent2', 'evaluator'
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class DialogState(TypedDict):
    """對話狀態"""

    input_query: NotRequired[str]  # 用戶初始輸入
    history: NotRequired[List[Message]]  # 對話歷史
    current_turn: NotRequired[int]  # 當前對話輪次
    max_turns: NotRequired[int]  # 最大對話輪次
    agent1_response: NotRequired[str]  # Agent 1 的回應
    agent2_response: NotRequired[str]  # Agent 2 的回應
    evaluator_notes: NotRequired[str]  # 評估者的筆記
    final_output: NotRequired[str]  # 最終輸出
    dialog_id: NotRequired[str]  # 對話ID
    use_streaming: NotRequired[bool]  # 是否使用串流模式
    performance_metrics: NotRequired[Dict[str, Dict[str, float]]]  # 性能指標


# --- 自訂 OpenAI LLM Runnable ---


class OpenAIRunnable(BaseModel):
    """封裝 OpenAI LLM 的 Langchain Runnable"""

    model: str = "gpt-4.1-nano-2025-04-14"  # 使用的模型名稱
    temperature: float = 0.7  # 溫度參數
    stream: bool = True  # 是否使用串流模式
    system_prompt: str = "你是一個專業的AI助手。"
    agent_name: str = "agent"  # Agent 名稱標識
    max_tokens: int = 1000  # 最大生成token數

    class Config:
        arbitrary_types_allowed = True

    async def _invoke_openai(
        self, state: Dict[str, Any], config: RunnableConfig | None = None
    ) -> Dict[str, Any]:
        """異步調用 OpenAI API"""
        # 開始計時 - wall clock time
        start_perf = perf_counter()
        # CPU時間
        start_process = process_time()

        performance_metrics = state.get("performance_metrics", {})
        node_metrics = {"start_perf": start_perf, "start_process": start_process}

        try:
            # 準備消息歷史
            history = state.get("history", [])
            use_streaming = state.get("use_streaming", True)

            logger.debug(
                f"調用 {self.agent_name}，對話歷史長度: {len(history)}，串流模式: {use_streaming}"
            )

            # 將對話歷史轉換為 LangChain 消息格式
            messages = [SystemMessage(content=self.system_prompt)]

            for msg in history:
                if msg.role == "user":
                    messages.append(HumanMessage(content=msg.content))
                else:
                    messages.append(AIMessage(content=msg.content))

            # 如果歷史為空，添加一個基本問候
            if len(messages) == 1:  # 只有系統提示
                if "input_query" in state and state["input_query"]:
                    messages.append(HumanMessage(content=state["input_query"]))
                else:
                    messages.append(
                        HumanMessage(content="您好，請問有什麼可以幫助您的嗎？")
                    )

            # 初始化 ChatOpenAI
            llm = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                streaming=use_streaming,
                api_key=api_key,
                max_tokens=self.max_tokens,
            )

            # 收集所有的輸出
            chunks = []
            full_content = ""

            # 標記API調用開始時間
            api_start_perf = perf_counter()
            api_start_process = process_time()
            node_metrics["api_start_perf"] = api_start_perf
            node_metrics["api_start_process"] = api_start_process

            # 開始處理
            if use_streaming:
                print(f"\n[{self.agent_name} 開始回應...]", flush=True)

                # 異步串流調用
                async for chunk in llm.astream(messages):
                    if chunk.content:
                        # 即時輸出
                        print(chunk.content, end="", flush=True)
                        chunks.append(chunk.content)

                # 構建完整回應
                full_content = "".join(chunks)
                print(f"\n[{self.agent_name} 回應完成]\n", flush=True)
            else:
                # 非串流模式
                print(f"\n[{self.agent_name} 處理中...]", flush=True)
                response = await llm.ainvoke(messages)
                full_content = response.content
                print(f"[{self.agent_name} 回應完成]\n", flush=True)

            # 標記API調用結束時間
            api_end_perf = perf_counter()
            api_end_process = process_time()
            node_metrics["api_end_perf"] = api_end_perf
            node_metrics["api_end_process"] = api_end_process
            node_metrics["api_duration_perf"] = api_end_perf - api_start_perf
            node_metrics["api_duration_process"] = api_end_process - api_start_process

            # 檢查回應是否為空
            if not full_content.strip():
                logger.warning(f"{self.agent_name} 回應為空，使用預設訊息")
                full_content = f"{self.agent_name} 未提供有效回應。"

            # 將回應添加到對話歷史
            new_message = Message(role=self.agent_name, content=full_content)
            updated_history = history + [new_message]

            # 節點執行結束時間
            end_perf = perf_counter()
            end_process = process_time()
            node_metrics["end_perf"] = end_perf
            node_metrics["end_process"] = end_process
            node_metrics["total_duration_perf"] = end_perf - start_perf
            node_metrics["total_duration_process"] = end_process - start_process

            # 性能數據
            performance_metrics[self.agent_name] = node_metrics

            # 記錄性能數據
            streaming_mode = "串流" if use_streaming else "非串流"
            perf_data.append(
                {
                    "節點": self.agent_name,
                    "模式": streaming_mode,
                    "wall_clock_time": node_metrics["total_duration_perf"],
                    "cpu_time": node_metrics["total_duration_process"],
                    "api_wall_clock_time": node_metrics["api_duration_perf"],
                    "api_cpu_time": node_metrics["api_duration_process"],
                    "時間戳": datetime.now().isoformat(),
                }
            )

            # 打印性能數據
            print(f"\n[{self.agent_name} 性能指標]")
            print(
                f"總執行時間 (wall clock): {node_metrics['total_duration_perf']:.4f}秒"
            )
            print(f"總 CPU 時間: {node_metrics['total_duration_process']:.4f}秒")
            print(
                f"API 調用時間 (wall clock): {node_metrics['api_duration_perf']:.4f}秒"
            )
            print(f"API 調用 CPU 時間: {node_metrics['api_duration_process']:.4f}秒")

            # 返回更新後的狀態
            return {
                f"{self.agent_name}_response": full_content,
                "history": updated_history,
                "performance_metrics": performance_metrics,
            }

        except Exception as e:
            logger.exception(f"{self.agent_name} 執行時發生錯誤\n{e}")

            # 節點執行結束時間（錯誤情況）
            end_perf = perf_counter()
            end_process = process_time()
            node_metrics["end_perf"] = end_perf
            node_metrics["end_process"] = end_process
            node_metrics["total_duration_perf"] = end_perf - start_perf
            node_metrics["total_duration_process"] = end_process - start_process
            node_metrics["error"] = str(e)

            performance_metrics[self.agent_name] = node_metrics

            # 返回錯誤訊息
            return {
                f"{self.agent_name}_response": f"很抱歉，處理您的請求時發生錯誤: {str(e)}",
                "error": str(e),
                "performance_metrics": performance_metrics,
            }

    def as_runnable(self) -> RunnableLambda:
        """將此類轉換為 RunnableLambda"""
        return RunnableLambda(self._invoke_openai)


# --- 對話流程節點 ---


async def initialize_dialog(state: Dict[str, Any]) -> Dict[str, Any]:
    """初始化對話"""
    # 開始計時
    start_perf = perf_counter()
    start_process = process_time()

    logger.info("初始化新對話")

    dialog_id = str(uuid.uuid4())
    current_time = datetime.now().isoformat()

    # 從輸入中提取用戶查詢和串流設置
    input_query = state.get("input_query", "您好，請問有什麼可以幫助您的嗎？")
    use_streaming = state.get("use_streaming", True)

    # 初始化用戶消息
    user_message = Message(role="user", content=input_query, timestamp=current_time)

    # 結束計時
    end_perf = perf_counter()
    end_process = process_time()

    # 性能指標
    performance_metrics = {
        "initialize": {
            "start_perf": start_perf,
            "start_process": start_process,
            "end_perf": end_perf,
            "end_process": end_process,
            "total_duration_perf": end_perf - start_perf,
            "total_duration_process": end_process - start_process,
        }
    }

    # 更新狀態
    return {
        "dialog_id": dialog_id,
        "history": [user_message],
        "current_turn": 1,
        "max_turns": 3,  # 設置最大對話輪次
        "use_streaming": use_streaming,
        "performance_metrics": performance_metrics,
    }


async def hotel_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """酒店預訂代理節點"""
    logger.info(f"執行酒店預訂代理，對話 ID: {state.get('dialog_id')}")

    # 創建酒店代理
    hotel_agent = OpenAIRunnable(
        model="gpt-4.1-nano-2025-04-14",
        temperature=0.3,
        stream=state.get("use_streaming", True),
        agent_name="hotel_agent",
        system_prompt="""你是一個專業的酒店預訂助手。
你的職責是協助客人了解酒店信息、提供房間建議、處理預訂請求，並回答與酒店相關的問題。
請保持專業、友善的態度，提供詳細而有用的回應。
如果客人詢問的問題超出你的知識範圍，請誠實告知並嘗試引導對話回到酒店預訂相關話題。

你可以提供的服務包括：
1. 酒店房型介紹和價格資訊
2. 預訂流程和政策說明
3. 酒店設施和服務說明
4. 入住和退房流程
5. 特殊要求處理

請記得遵循以下溝通原則：
- 使用禮貌、專業的語言
- 提供具體、實用的資訊
- 主動了解客人需求
- 適時提出建議
- 確認重要資訊""",
    )

    # 執行酒店代理
    result = await hotel_agent.as_runnable().ainvoke(state)
    logger.info(f"酒店預訂代理執行完成，對話 ID: {state.get('dialog_id')}")

    return result


async def travel_agent_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """旅遊顧問代理節點"""
    logger.info(f"執行旅遊顧問代理，對話 ID: {state.get('dialog_id')}")

    # 創建旅遊代理
    travel_agent = OpenAIRunnable(
        model="gpt-4.1-nano-2025-04-14",
        temperature=0.5,
        stream=state.get("use_streaming", True),
        agent_name="travel_agent",
        system_prompt="""你是一個專業的旅遊顧問。
你的職責是協助客人規劃旅行，提供目的地建議、景點推薦、當地文化和習俗資訊、交通建議等。
請保持友善和專業的態度，提供詳細而有價值的旅遊建議。

你可以提供的服務包括：
1. 目的地推薦和介紹
2. 景點和活動建議
3. 當地美食和餐廳介紹
4. 交通和行程規劃建議
5. 旅行安全和健康提示
6. 當地文化和習俗資訊

請記得遵循以下溝通原則：
- 提供個性化的建議
- 考慮客人的偏好和需求
- 提供實用的旅行技巧
- 分享當地獨特體驗
- 幫助客人做出明智的旅行決策

如果客人詢問超出你知識範圍的問題，請誠實說明並引導對話回到旅遊相關話題。""",
    )

    # 執行旅遊代理
    result = await travel_agent.as_runnable().ainvoke(state)
    logger.info(f"旅遊顧問代理執行完成，對話 ID: {state.get('dialog_id')}")

    return result


async def evaluator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """評估者節點"""
    # 開始計時
    start_perf = perf_counter()
    start_process = process_time()

    node_metrics = {"start_perf": start_perf, "start_process": start_process}

    performance_metrics = state.get("performance_metrics", {})

    logger.info(f"執行評估者，對話 ID: {state.get('dialog_id')}")

    # 提取酒店代理和旅遊代理的回應
    hotel_response = state.get("hotel_agent_response", "")
    travel_response = state.get("travel_agent_response", "")

    # 構建輸入
    evaluation_input = f"""
酒店預訂代理回應:
{hotel_response}

旅遊顧問代理回應:
{travel_response}
"""

    # 創建評估者
    evaluator = OpenAIRunnable(
        model="gpt-4.1-nano-2025-04-14",
        temperature=0.3,
        stream=state.get("use_streaming", True),
        agent_name="evaluator",
        system_prompt="""你是一個專業的對話評估專家。
你的任務是評估兩個AI代理的回應，並提供以下分析：
1. 兩個代理各自的優點和不足
2. 哪個代理的回應更能滿足用戶需求及原因
3. 如何結合兩個代理的優點，提供最佳回應
4. 具體改進建議

請基於以下標準進行評估：
- 回應的相關性和針對性
- 資訊的準確性和完整性
- 語言表達的清晰度和專業性
- 對用戶需求的理解和滿足程度
- 整體服務體驗

請提供公正、客觀的評估，並給出具體的分析和建議。""",
    )

    # 更新狀態，添加評估輸入
    eval_state = {**state, "input_query": evaluation_input}

    # API調用開始時間
    api_start_perf = perf_counter()
    api_start_process = process_time()
    node_metrics["api_start_perf"] = api_start_perf
    node_metrics["api_start_process"] = api_start_process

    # 執行評估者
    result = await evaluator.as_runnable().ainvoke(eval_state)

    # 合併結果
    merged_result = {**result}

    # 如果原始性能指標存在，保留它們
    if "performance_metrics" in state:
        merged_result["performance_metrics"] = {
            **state["performance_metrics"],
            **result.get("performance_metrics", {}),
        }

    logger.info(f"評估者執行完成，對話 ID: {state.get('dialog_id')}")

    return merged_result


async def summarize_dialog(state: Dict[str, Any]) -> Dict[str, Any]:
    """總結對話並保存結果"""
    # 開始計時
    start_perf = perf_counter()
    start_process = process_time()

    logger.info(f"總結對話，對話 ID: {state.get('dialog_id')}")

    dialog_id = state.get("dialog_id", str(uuid.uuid4()))
    hotel_response = state.get("hotel_agent_response", "")
    travel_response = state.get("travel_agent_response", "")
    evaluator_response = state.get("evaluator_response", "")
    history = state.get("history", [])
    use_streaming = state.get("use_streaming", True)
    performance_metrics = state.get("performance_metrics", {})

    # 構建總結
    stream_mode = "串流模式" if use_streaming else "非串流模式"

    # 計算性能統計
    perf_stats = "\n===== 性能統計 =====\n"
    for node_name, metrics in performance_metrics.items():
        if node_name == "initialize":
            continue
        perf_stats += f"\n--- {node_name} ---\n"
        perf_stats += (
            f"總執行時間 (wall clock): {metrics.get('total_duration_perf', 0):.4f}秒\n"
        )
        perf_stats += f"總 CPU 時間: {metrics.get('total_duration_process', 0):.4f}秒\n"
        if "api_duration_perf" in metrics:
            perf_stats += f"API 調用時間 (wall clock): {metrics.get('api_duration_perf', 0):.4f}秒\n"
            perf_stats += (
                f"API 調用 CPU 時間: {metrics.get('api_duration_process', 0):.4f}秒\n"
            )

    summary = f"""
對話 ID: {dialog_id}
模式: {stream_mode}
時間: {datetime.now().isoformat()}

{perf_stats}

===== 對話歷史 =====
{json.dumps([msg.dict() for msg in history], ensure_ascii=False, indent=2)}

===== 酒店預訂代理回應 =====
{hotel_response}

===== 旅遊顧問代理回應 =====
{travel_response}

===== 評估者分析 =====
{evaluator_response}
"""

    # 保存對話結果
    output_file = output_dir / f"dialog_{dialog_id}_{stream_mode}.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info(f"對話結果已保存到: {output_file}")

    # 結束計時
    end_perf = perf_counter()
    end_process = process_time()

    # 更新性能指標
    performance_metrics["summarize"] = {
        "start_perf": start_perf,
        "start_process": start_process,
        "end_perf": end_perf,
        "end_process": end_process,
        "total_duration_perf": end_perf - start_perf,
        "total_duration_process": end_process - start_process,
    }

    # 更新整體性能指標
    overall_start = min(
        [
            metrics.get("start_perf", float("inf"))
            for metrics in performance_metrics.values()
        ]
    )
    overall_end = max(
        [metrics.get("end_perf", 0) for metrics in performance_metrics.values()]
    )
    overall_duration = overall_end - overall_start

    # 打印整體性能
    print("\n===== 整體性能 =====")
    print(f"總執行時間 (wall clock): {overall_duration:.4f}秒")
    print(f"模式: {stream_mode}")

    # 返回最終結果
    return {
        "final_output": f"對話已完成並保存到: {output_file}",
        "summary": summary,
        "performance_metrics": performance_metrics,
        "overall_duration": overall_duration,
    }


# --- 流程控制 ---


def build_graph() -> StateGraph:
    """構建 Langgraph 流程圖"""
    workflow = StateGraph(FlowState)

    # 使用串流模式
    raccoon_agent_2 = ChatOpenAI(
        model="gpt-4-1106-nano", temperature=0.7, streaming=True
    )
    raccoon_agent_3 = ChatOpenAI(
        model="gpt-4-1106-nano", temperature=0.7, streaming=True
    )

    # 偏函數應用，將 agent_id 和 runnable 綁定到節點函數
    # 使用 functools.partial 替代 lambda 以提高效能 (雖然此處差異不大)
    from functools import partial

    llm_agent_2_node_partial = partial(
        llm_agent_node, agent_id=137, raccoon_runnable=raccoon_agent_2.as_runnable()
    )
    llm_agent_3_node_partial = partial(
        llm_agent_node, agent_id=137, raccoon_runnable=raccoon_agent_3.as_runnable()
    )

    workflow.add_node("llm_agent_2", llm_agent_2_node_partial)
    workflow.add_node("llm_agent_3", llm_agent_3_node_partial)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("final_llm", final_llm_node)

    # 設置一個虛擬的起始分支節點，確保並行執行
    def start_branch(state: FlowState):
        # 這個節點不做任何事，只是為了觸發後續的並行分支
        logger.info("流程開始，準備分支到 LLM Agent 2 和 3")
        return {}  # 返回空字典，不修改狀態

    workflow.add_node("start_branch", start_branch)
    workflow.set_entry_point("start_branch")
    workflow.add_edge("start_branch", "llm_agent_2")
    workflow.add_edge("start_branch", "llm_agent_3")

    # 添加聚合和最終節點的邊
    workflow.add_edge("llm_agent_3", "aggregator")  # Agent 3 完成後也到 aggregator
    workflow.add_edge("aggregator", "final_llm")

    # 添加結束點
    workflow.add_edge("final_llm", END)

    # 編譯流程圖
    graph = workflow.compile()
    logger.info("對話流程圖構建完成")

    return graph


# --- 主執行函數 ---


async def run_dialog(user_query: str, use_streaming: bool = True) -> Dict[str, Any]:
    """執行對話流程"""
    # 全局計時開始
    global_start_perf = perf_counter()
    global_start_process = process_time()

    logger.info(f"開始執行對話流程，使用{'串流' if use_streaming else '非串流'}模式")
    print(f"\n===== 開始執行{'串流' if use_streaming else '非串流'}模式對話 =====")

    # 構建流程圖
    graph = build_dialog_graph()

    # 初始狀態
    initial_state: DialogState = {
        "input_query": user_query,
        "use_streaming": use_streaming,
    }

    logger.info(f"初始查詢: {user_query}")

    # 執行流程
    try:
        # 執行對話流程
        final_state = await graph.ainvoke(initial_state)
        logger.info("對話流程執行完畢")

        # 全局計時結束
        global_end_perf = perf_counter()
        global_end_process = process_time()
        total_perf_time = global_end_perf - global_start_perf
        total_process_time = global_end_process - global_start_process

        # 打印全局性能
        print("\n===== 全局性能 =====")
        print(f"總執行時間 (wall clock): {total_perf_time:.4f}秒")
        print(f"總 CPU 時間: {total_process_time:.4f}秒")

        # 記錄全局性能
        perf_data.append(
            {
                "節點": "全局",
                "模式": "串流" if use_streaming else "非串流",
                "wall_clock_time": total_perf_time,
                "cpu_time": total_process_time,
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
                "時間戳": datetime.now().isoformat(),
            }
        )

        # 保存性能數據到CSV
        save_performance_data()

        # 輸出摘要
        print("\n===== 對話摘要 =====")
        if "summary" in final_state:
            show_summary_highlights(final_state["summary"])

        return final_state

    except Exception as e:
        logger.exception(f"對話流程執行時發生錯誤: {e}")
        return {"error": str(e), "final_output": f"對話流程執行失敗: {str(e)}"}


def save_performance_data() -> None:
    """將性能數據儲存到CSV檔案"""
    if not perf_data:
        logger.warning("沒有性能數據可儲存")
        return

    # 建立檔案名稱
    csv_file = (
        output_dir
        / f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    # 定義欄位名稱
    fieldnames = [
        "節點",
        "模式",
        "wall_clock_time",
        "cpu_time",
        "api_wall_clock_time",
        "api_cpu_time",
        "時間戳",
    ]

    try:
        # 寫入CSV檔案
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for data in perf_data:
                writer.writerow(data)

        logger.info(f"性能數據已儲存到: {csv_file}")
        print(f"\n性能數據已儲存到: {csv_file}")
    except Exception as e:
        logger.exception(f"儲存性能數據時發生錯誤: {e}")
        print(f"儲存性能數據時發生錯誤: {e}")


def show_summary_highlights(summary: str) -> None:
    """顯示對話摘要的重點部分"""
    lines = summary.strip().split("\n")

    # 找出性能統計部分
    perf_stats_start = -1
    perf_stats_end = -1

    for i, line in enumerate(lines):
        if "===== 性能統計 =====" in line:
            perf_stats_start = i
        elif perf_stats_start != -1 and "===== 對話歷史 =====" in line:
            perf_stats_end = i
            break

    # 顯示頭部資訊 (前10行)
    print("--- 對話摘要開頭 ---")
    for i in range(min(10, len(lines))):
        print(lines[i])

    # 顯示性能統計部分
    if perf_stats_start != -1 and perf_stats_end != -1:
        print("\n--- 性能統計 ---")
        for i in range(perf_stats_start, perf_stats_end):
            print(lines[i])

    print("\n(完整對話摘要已儲存到檔案)")


# 如果是直接執行此檔案（非導入）
if __name__ == "__main__":
    # 為 Windows 系統設置合適的事件循環政策
    if platform.system() == "Windows":
        logger.info("在 Windows 上設置 WindowsSelectorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # 在非 Windows 系統上使用 uvloop (如果可用)
    elif has_uvloop:
        try:
            uvloop.install()
            logger.info("使用 uvloop 加速異步操作")
        except Exception as e:
            logger.warning(f"無法安裝 uvloop: {e}")
    else:
        logger.info("使用標準事件循環")

    # 執行對話流程測試
    test_query = "你好，我計劃去台北旅行，並且需要預訂酒店，有什麼推薦嗎？"

    # 先執行串流模式
    asyncio.run(run_dialog(test_query, use_streaming=True))

    # 然後執行非串流模式
    asyncio.run(run_dialog(test_query, use_streaming=False))
