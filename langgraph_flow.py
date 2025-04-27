# langgraph_flow.py

import asyncio
import csv
import os
import platform
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Literal, NotRequired, TypedDict

# 使用try-except避免Windows系統錯誤
try:
    import uvloop

    has_uvloop = True
except ImportError:
    has_uvloop = False

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger

# --- 配置與初始化 ---

# 配置日誌
logger.add("langgraph_flow.log", rotation="10 MB", retention="7 days", level="DEBUG")

# 加載環境變數 (例如 OPENAI_API_KEY)
load_dotenv()

# 檢查 OpenAI API Key 是否設置
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY 環境變數未設置，OpenAI LLM 可能無法工作。")
    raise ValueError("OPENAI_API_KEY environment variable not set.")  # 或者直接報錯

# --- Langgraph 狀態定義 ---


class FlowState(TypedDict):
    source_input: NotRequired[str]
    llm2_output: NotRequired[str]
    llm3_output: NotRequired[str]
    merged_output: NotRequired[str]
    final_output: NotRequired[str]


# --- Langgraph 節點定義 ---


async def llm_agent_node(
    state: FlowState, agent_id: int, llm_runnable, system_prompt=None
) -> dict[str, Any]:
    """
    執行 LLM Agent 節點（OpenAI LLM），並個別計時。
    可自訂 system_prompt。
    """
    logger.info(f"進入 LLM Agent {agent_id} 節點 (OpenAI)")
    message = state["source_input"]
    is_lambda_mode = state.get("執行模式") == "Lambda串流模式"
    node_start_time = perf_counter()  # 紀錄節點開始時間

    if not message:
        logger.warning(f"LLM Agent {agent_id}: 原始訊息為空")
        return {f"llm_{agent_id}_output": {"error": "Original message is empty."}}

    # 獲取執行模式和串流模式
    exec_mode = state.get("執行模式", "未知")
    streaming_mode = "串流" if state.get("use_streaming", False) else "非串流"

    # 計時開始
    start_perf = perf_counter()
    start_process = process_time()
    node_metrics = {
        "agent_id": agent_id,
        "llm_type": "openai",
        "start_perf": start_perf,
        "start_process": start_process,
    }

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        if system_prompt is None:
            system_prompt = (
                "你是專業客服 AI，請根據用戶需求給予簡潔、禮貌且專業的回應。"
            )
        system_message = SystemMessage(content=system_prompt)
        human_message = HumanMessage(content=message)

        first_token_received = False
        first_token_time = None

        # 對於串流模式，我們需要捕獲第一個token的時間
        if state.get("use_streaming", False) and hasattr(llm_runnable, "astream"):
            # 直接使用 astream 並手動捕獲第一個 token
            response_chunks = []
            async for chunk in llm_runnable.astream([system_message, human_message]):
                if chunk.content and not first_token_received:
                    first_token_time = perf_counter()
                    first_token_received = True
                    # 記錄 Lambda 節點執行時間
                    if is_lambda_mode:
                        node_duration = first_token_time - node_start_time
                        lambda_node_timings[f"Agent_{agent_id}"] = node_duration
                        logger.info(
                            f"Lambda模式: Agent {agent_id} 節點執行時間: {node_duration:.4f}秒"
                        )
                response_chunks.append(chunk)
            response = response_chunks[-1] if response_chunks else None
        else:
            # 非串流模式，使用普通調用
            response = await llm_runnable.ainvoke([system_message, human_message])
            # 對於Lambda模式，記錄執行時間
            if is_lambda_mode:
                node_end_time = perf_counter()
                node_duration = node_end_time - node_start_time
                lambda_node_timings[f"Agent_{agent_id}"] = node_duration
                logger.info(
                    f"Lambda模式: Agent {agent_id} 節點執行時間: {node_duration:.4f}秒"
                )

        content = response

        # 計時結束
        end_perf = perf_counter()
        end_process = process_time()
        node_metrics["end_perf"] = end_perf
        node_metrics["end_process"] = end_process
        node_metrics["duration_perf"] = end_perf - start_perf
        node_metrics["duration_process"] = end_process - start_process

        # 記錄性能
        perf_data.append(
            {
                "執行模式": exec_mode,
                "節點": f"Agent_{agent_id}",
                "llm_type": "openai",
                "串流模式": streaming_mode,
                "wall_clock_time": node_metrics["duration_perf"],
                "cpu_time": node_metrics["duration_process"],
                "api_wall_clock_time": node_metrics["duration_perf"],
                "api_cpu_time": node_metrics["duration_process"],
                "input": message,
                "output": content,
                "時間戳": datetime.now().isoformat(),
            }
        )
        logger.info(f"LLM Agent {agent_id} (openai) 執行完成")
        # 關鍵修正：回傳 key 必須為 llm2_output 或 llm3_output
        key = f"llm{agent_id}_output"
        return {key: content}
    except Exception as e:
        end_perf = perf_counter()
        end_process = process_time()
        node_metrics["end_perf"] = end_perf
        node_metrics["end_process"] = end_process
        node_metrics["duration_perf"] = end_perf - start_perf
        node_metrics["duration_process"] = end_process - start_process
        node_metrics["error"] = str(e)
        perf_data.append(
            {
                "執行模式": exec_mode,
                "節點": f"Agent_{agent_id}",
                "llm_type": "openai",
                "串流模式": streaming_mode,
                "wall_clock_time": node_metrics["duration_perf"],
                "cpu_time": node_metrics["duration_process"],
                "api_wall_clock_time": node_metrics["duration_perf"],
                "api_cpu_time": node_metrics["duration_process"],
                "input": message,
                "output": f"error: {str(e)}",
                "時間戳": datetime.now().isoformat(),
                "error": str(e),
            }
        )
        logger.exception(f"LLM Agent {agent_id} (openai) 執行時發生錯誤")
        key = f"llm{agent_id}_output"
        return {key: {"error": f"Agent {agent_id} (openai) failed: {str(e)}"}}


async def aggregator_node(state: FlowState) -> dict[str, Any]:
    """聚合 LLM Agent 2 和 3 結果的節點"""
    logger.info("進入 Aggregator 節點")
    is_lambda_mode = state.get("執行模式") == "Lambda串流模式"
    node_start_time = perf_counter()  # 紀錄節點開始時間

    # 獲取執行模式和串流模式
    exec_mode = state.get("執行模式", "未知")
    streaming_mode = "串流" if state.get("use_streaming", False) else "非串流"

    llm_2_resp = state.get("llm2_output")
    llm_3_resp = state.get("llm3_output")

    # 提取內容
    content_2 = ""
    if hasattr(llm_2_resp, "content"):
        content_2 = getattr(llm_2_resp, "content", "")
    elif isinstance(llm_2_resp, dict) and "content" in llm_2_resp:
        content_2 = llm_2_resp["content"]

    content_3 = ""
    if hasattr(llm_3_resp, "content"):
        content_3 = getattr(llm_3_resp, "content", "")
    elif isinstance(llm_3_resp, dict) and "content" in llm_3_resp:
        content_3 = llm_3_resp["content"]

    # 檢查是否有內容
    if not content_2.strip() and not content_3.strip():
        logger.warning("兩個 Agent 的回應均為空")
        merged_output = "兩個 Agent 均未返回有效回應。請檢查連接或嘗試不同的查詢。"
    else:
        merged_output = (
            f"來自 Agent 2 的觀點:\n{content_2}\n\n來自 Agent 3 的觀點:\n{content_3}"
        )

    logger.info("Aggregator 節點執行完成")
    logger.debug(f"聚合後內容: {merged_output[:100]}...")

    # 計算節點執行時間
    node_duration = perf_counter() - node_start_time

    # 對於Lambda模式，記錄節點執行時間
    if is_lambda_mode:
        lambda_node_timings["Aggregator"] = node_duration
        logger.info(f"Lambda模式: Aggregator 節點執行時間: {node_duration:.4f}秒")

    # 記錄 input/output
    perf_data.append(
        {
            "執行模式": exec_mode,
            "節點": "Aggregator",
            "llm_type": "N/A",
            "串流模式": streaming_mode,
            "wall_clock_time": node_duration,
            "cpu_time": process_time(),
            "api_wall_clock_time": node_duration,
            "api_cpu_time": process_time(),
            "input": {"llm2_output": llm_2_resp, "llm3_output": llm_3_resp},
            "output": merged_output,
            "時間戳": datetime.now().isoformat(),
        }
    )
    return {"merged_output": merged_output}


async def final_llm_node(state: FlowState) -> dict[Literal["final_output"], str]:
    """執行最終 LLM Agent (OpenAI) 並串流輸出的節點"""
    logger.info("進入 Final LLM 節點")
    merged_output = state.get("merged_output")
    is_lambda_mode = state.get("執行模式") == "Lambda串流模式"
    lambda_global_start = state.get("lambda_global_start_time", None)
    node_start_time = perf_counter()  # 記錄節點開始時間

    # 獲取執行模式和串流模式
    exec_mode = state.get("執行模式", "未知")
    streaming_mode = "串流" if state.get("use_streaming", False) else "非串流"

    if state.get("error"):  # 檢查上游是否有錯誤
        logger.warning(f"由於上游錯誤，跳過 Final LLM 節點: {state['error']}")
        return {"final_output": "流程執行中發生錯誤"}

    if not merged_output:
        logger.warning("Final LLM: 聚合內容為空")
        return {"final_output": "聚合內容為空，無法生成最終回應。"}

    logger.debug(f"Final LLM 收到的聚合內容: {merged_output[:150]}...")

    if (
        merged_output.strip() == "來自 Agent 2 的觀點:\n\n\n來自 Agent 3 的觀點:\n"
        or len(merged_output.strip()) < 50
    ):
        logger.warning("聚合內容缺少實質內容，使用預設訊息替代")
        merged_output = """來自 Agent 2 的觀點:
這是一個測試對話，我會根據評估標準給您提供客服對話的專業評分和建議。

來自 Agent 3 的觀點:
聚合內容似乎缺少實質內容，可能是由於連接問題或API回應異常。建議檢查API連接和stream_chat方法的實現。
"""

    from prompt import basic_evaluator_prompt

    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=True)
    system_message = SystemMessage(content=basic_evaluator_prompt)
    human_message = HumanMessage(content=merged_output)

    try:
        print("\n--- 評估結果 ---\n", flush=True)
        output_chunks = []
        first_token_received = False
        first_token_time = None
        node_duration = None

        async for chunk in llm.astream([system_message, human_message]):
            if chunk.content:
                # 捕獲第一個token的時間
                if not first_token_received:
                    first_token_time = perf_counter()
                    first_token_received = True

                    # 如果是Lambda模式，則在這裡記錄從開始到第一個token的時間
                    if is_lambda_mode and lambda_global_start:
                        lambda_response_time = first_token_time - lambda_global_start

                        # 計算 Final LLM 節點執行時間
                        if is_lambda_mode:
                            node_duration = first_token_time - node_start_time
                            lambda_node_timings["Final_LLM"] = node_duration
                            logger.info(
                                f"Lambda模式: Final LLM 節點執行時間: {node_duration:.4f}秒"
                            )

                        logger.info(
                            f"Lambda 模式首個 token 響應時間: {lambda_response_time:.4f}秒"
                        )
                        print(
                            f"\n--- Lambda 首個 token 響應時間: {lambda_response_time:.4f}秒 ---"
                        )

                        # 記錄 Lambda 首個 token 響應時間
                        perf_data.append(
                            {
                                "執行模式": "Lambda串流模式",
                                "節點": "首個Token",
                                "llm_type": "openai",
                                "串流模式": "串流",
                                "wall_clock_time": lambda_response_time,
                                "cpu_time": process_time()
                                - state.get("lambda_global_start_cpu", 0),
                                "api_wall_clock_time": lambda_response_time,
                                "api_cpu_time": process_time()
                                - state.get("lambda_global_start_cpu", 0),
                                "時間戳": datetime.now().isoformat(),
                            }
                        )

                # 正常輸出token
                print(chunk.content, end="", flush=True)
                output_chunks.append(chunk.content)

        print("\n", flush=True)
        logger.info("Final LLM 串流輸出完成")

        # 如果沒有計算過節點時間，則在此計算
        if node_duration is None:
            node_duration = perf_counter() - node_start_time

        # 記錄 input/output
        perf_data.append(
            {
                "執行模式": exec_mode,
                "節點": "Final_LLM",
                "llm_type": "openai",
                "串流模式": streaming_mode,
                "wall_clock_time": node_duration,
                "cpu_time": process_time(),
                "api_wall_clock_time": node_duration,
                "api_cpu_time": process_time(),
                "input": merged_output,
                "output": "".join(output_chunks),
                "時間戳": datetime.now().isoformat(),
            }
        )
        return {"final_output": "done"}

    except Exception as e:
        logger.exception("Final LLM 執行時發生錯誤")

        # 記錄異常情況
        node_duration = perf_counter() - node_start_time
        perf_data.append(
            {
                "執行模式": exec_mode,
                "節點": "Final_LLM",
                "llm_type": "openai",
                "串流模式": streaming_mode,
                "wall_clock_time": node_duration,
                "cpu_time": process_time(),
                "api_wall_clock_time": node_duration,
                "api_cpu_time": process_time(),
                "input": merged_output,
                "output": f"error: {str(e)}",
                "時間戳": datetime.now().isoformat(),
                "error": str(e),
            }
        )
        return {"final_output": f"Final LLM failed: {str(e)}"}


# --- Langgraph 圖構建 ---


def build_graph(use_streaming=True) -> StateGraph:
    """構建 Langgraph 流程圖（OpenAI LLM only）"""
    workflow = StateGraph(FlowState)
    from functools import partial

    llm2 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
    )
    llm3 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
    )

    # 專業酒店預訂助手 system_prompt
    hotel_system_prompt = """你是一個專業的酒店預訂助手。
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
- 確認重要資訊"""

    llm_agent_2_node_partial = partial(
        llm_agent_node, agent_id=2, llm_runnable=llm2, system_prompt=hotel_system_prompt
    )
    llm_agent_3_node_partial = partial(
        llm_agent_node, agent_id=3, llm_runnable=llm3, system_prompt=hotel_system_prompt
    )

    workflow.add_node("llm_agent_2", llm_agent_2_node_partial)
    workflow.add_node("llm_agent_3", llm_agent_3_node_partial)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("final_llm", final_llm_node)

    def start_branch(state: FlowState):
        logger.info("流程開始，準備分支到 LLM Agent 2 和 3")
        return {}

    workflow.add_node("start_branch", start_branch)
    workflow.set_entry_point("start_branch")
    workflow.add_edge("start_branch", "llm_agent_2")
    workflow.add_edge("start_branch", "llm_agent_3")
    workflow.add_edge("llm_agent_3", "aggregator")
    workflow.add_edge("aggregator", "final_llm")
    workflow.add_edge("final_llm", END)

    app = workflow.compile()
    logger.info("Langgraph 圖構建完成 (OpenAI LLM only)")
    return app


# --- 性能監控變數 ---

perf_data = []
lambda_node_timings = {}  # 用於記錄Lambda各節點的執行時間
output_dir = Path("performance_outputs")
output_dir.mkdir(exist_ok=True)


def save_performance_data() -> None:
    """將性能數據儲存到CSV檔案"""
    if not perf_data:
        logger.warning("沒有性能數據可儲存")
        return

    csv_file = (
        output_dir
        / f"parallel_vs_sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    fieldnames = [
        "執行模式",
        "節點",
        "llm_type",
        "串流模式",
        "wall_clock_time",
        "cpu_time",
        "api_wall_clock_time",
        "api_cpu_time",
        "input",
        "output",
        "時間戳",
        "error",
        "節點佔比",
    ]

    try:
        # 先確定執行模式，通過檢查包含 Agent_2、Agent_3 等節點的數據
        exec_modes = set()
        stream_modes = set()
        node_types = set()

        for entry in perf_data:
            if entry.get("執行模式") and entry["執行模式"] != "未知":
                exec_modes.add(entry["執行模式"])
            if entry.get("串流模式") and entry["串流模式"] != "未知":
                stream_modes.add(entry["串流模式"])
            if entry.get("節點"):
                node_types.add(entry["節點"])

        # 確定最可能的執行模式
        most_likely_exec_mode = None
        if "Lambda轉出串流" in exec_modes:
            most_likely_exec_mode = "Lambda串流模式"
        elif "並行執行" in exec_modes:
            most_likely_exec_mode = "並行模式"
        elif "順序執行" in exec_modes:
            most_likely_exec_mode = "順序模式"
        elif (
            "Agent_2" in node_types
            and "Agent_3" in node_types
            and "Aggregator" in node_types
        ):
            # 根據節點組合推斷執行模式
            most_likely_exec_mode = "並行模式"

        # 確定最可能的串流模式
        most_likely_stream_mode = None
        if "串流" in stream_modes:
            most_likely_stream_mode = "串流"
        elif "非串流" in stream_modes:
            most_likely_stream_mode = "非串流"

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            # 添加節點佔比計算
            lambda_node_total = calculate_lambda_total_node_time()

            for row in perf_data:
                # 將 input/output 轉為字串避免 CSV 格式錯誤
                row = dict(row)
                if isinstance(row.get("input"), (dict, list)):
                    row["input"] = str(row["input"])
                if isinstance(row.get("output"), (dict, list)):
                    row["output"] = str(row["output"])

                # 確保執行模式有值
                if not row.get("執行模式") or row["執行模式"] == "未知":
                    if "Lambda" in row.get("節點", "") or row["節點"] == "首個Token":
                        row["執行模式"] = "Lambda串流模式"
                    elif most_likely_exec_mode:
                        row["執行模式"] = most_likely_exec_mode

                # 確保串流模式有值
                if not row.get("串流模式") or row["串流模式"] == "未知":
                    if row.get("執行模式") == "Lambda串流模式":
                        row["串流模式"] = "串流"
                    elif most_likely_stream_mode:
                        row["串流模式"] = most_likely_stream_mode

                # 將 N/A 轉為空字串或者 0.0
                for key in [
                    "wall_clock_time",
                    "cpu_time",
                    "api_wall_clock_time",
                    "api_cpu_time",
                ]:
                    if row.get(key) == "N/A":
                        # 如果是計時相關欄位，轉為 0.0
                        row[key] = 0.0

                # 計算Lambda節點佔比
                if row["執行模式"] == "Lambda串流模式":
                    node_name = row["節點"]
                    if node_name in lambda_node_timings and lambda_node_total > 0:
                        row["節點佔比"] = (
                            f"{lambda_node_timings[node_name]/lambda_node_total*100:.2f}%"
                        )

                writer.writerow(row)

        logger.info(f"性能數據已儲存到: {csv_file}")
        print(f"\n性能數據已儲存到: {csv_file}")
    except Exception as e:
        logger.exception(f"儲存性能數據時發生錯誤: {e}")
        print(f"儲存性能數據時發生錯誤: {e}")


def calculate_lambda_total_node_time():
    """計算 Lambda 模式下所有節點的總執行時間"""
    if not lambda_node_timings:
        return 0.0

    total_time = sum(lambda_node_timings.values())
    return total_time


def analyze_performance_data():
    """分析性能數據並顯示比較結果"""
    if not perf_data:
        print("沒有性能數據可分析")
        return

    grouped_data = {}
    for entry in perf_data:
        # 確保Lambda模式也被正確處理
        if entry["節點"] == "全局" or entry["節點"] == "Lambda主線程":
            key = (entry["執行模式"], entry.get("llm_type", ""), entry["串流模式"])
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(entry)

    print("\n===== 各種模式執行時間對比 =====")
    print(
        f"{'執行模式':<16} | {'llm_type':<8} | {'串流模式':<8} | {'總執行時間(秒)':<15}"
    )
    print("-" * 65)

    for (exec_mode, llm_type, stream_mode), entries in grouped_data.items():
        avg_time = sum(entry["wall_clock_time"] for entry in entries) / len(entries)
        print(f"{exec_mode:<16} | {llm_type:<8} | {stream_mode:<8} | {avg_time:<15.4f}")

    # 添加Lambda轉出效率比較
    lambda_entries = [
        e
        for e in perf_data
        if e["執行模式"] == "Lambda串流模式" and e["節點"] == "首個Token"
    ]

    lambda_main_entries = [
        e
        for e in perf_data
        if e["執行模式"] == "Lambda串流模式" and e["節點"] == "Lambda主線程"
    ]

    if lambda_entries or lambda_main_entries:
        print("\n===== Lambda模式效率分析 =====")

        # 使用首個Token時間或總執行時間
        if lambda_entries:
            lambda_time = lambda_entries[0]["wall_clock_time"]
            print(f"Lambda首個Token響應時間: {lambda_time:.4f}秒")
        elif lambda_main_entries:
            lambda_time = lambda_main_entries[0]["wall_clock_time"]
            print(f"Lambda總執行時間: {lambda_time:.4f}秒 (無首個Token記錄)")
        else:
            lambda_time = 0
            print("未能找到Lambda執行時間記錄")

        # 計算Lambda節點總執行時間
        lambda_node_total_time = calculate_lambda_total_node_time()
        print(f"Lambda節點總執行時間: {lambda_node_total_time:.6f}秒")

        if lambda_node_timings:
            print("\n各節點實際執行時間明細:")
            for node, duration in lambda_node_timings.items():
                print(
                    f"  {node}: {duration:.6f}秒 ({duration/lambda_node_total_time*100:.2f}%)"
                )

        # 計算Lambda模式效率比
        if lambda_node_total_time > 0 and lambda_time > 0:
            efficiency_ratio = lambda_time / lambda_node_total_time
            print(f"\nLambda端到端時間與節點執行時間比例: {efficiency_ratio:.2f}倍")
            if efficiency_ratio > 1:
                print(
                    f"這表示約有 {(efficiency_ratio-1)*100:.1f}% 的時間用於網路傳輸和其他開銷"
                )
            else:
                print("節點執行時間總和大於端到端回應時間，可能是由於並行處理效應")

        # 與其他模式比較 (只有有效的lambda_time才比較)
        if lambda_time > 0:
            print("\n與其他執行模式比較:")
            for (exec_mode, llm_type, stream_mode), entries in grouped_data.items():
                if exec_mode != "Lambda轉出串流":
                    avg_time = sum(entry["wall_clock_time"] for entry in entries) / len(
                        entries
                    )
                    speedup = avg_time / lambda_time
                    print(f"  比{exec_mode}快 {speedup:.2f}倍")


# --- 主執行函數 ---


async def run_sequential(initial_message: str, use_streaming: bool = True):
    """執行順序流程（非並行）"""
    global_start_perf = perf_counter()
    global_start_process = process_time()

    mode = "串流" if use_streaming else "非串流"
    logger.info(f"開始執行【順序流程】，使用{mode}模式")
    print(f"\n===== 開始執行【順序流程】({mode}) =====")

    workflow = StateGraph(FlowState)
    from functools import partial

    llm2 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
    )
    llm3 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
    )

    llm_agent_2_node_partial = partial(llm_agent_node, agent_id=2, llm_runnable=llm2)
    llm_agent_3_node_partial = partial(llm_agent_node, agent_id=3, llm_runnable=llm3)

    workflow.add_node("start", lambda x: x)
    workflow.add_node("llm_agent_2", llm_agent_2_node_partial)
    workflow.add_node("llm_agent_3", llm_agent_3_node_partial)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("final_llm", final_llm_node)
    workflow.set_entry_point("start")
    workflow.add_edge("start", "llm_agent_2")
    workflow.add_edge("llm_agent_2", "llm_agent_3")
    workflow.add_edge("llm_agent_3", "aggregator")
    workflow.add_edge("aggregator", "final_llm")
    workflow.add_edge("final_llm", END)
    app = workflow.compile()
    logger.info("順序執行流程圖構建完成 (OpenAI LLM only)")

    initial_state: FlowState = {
        "source_input": initial_message,
        "use_streaming": mode,
        "執行模式": "順序模式",
        "llm_type": "openai",
    }
    logger.info(f"初始查詢: {initial_message}")

    try:
        final_state = await app.ainvoke(initial_state)
        logger.info("順序流程執行完畢")

        global_end_perf = perf_counter()
        global_end_process = process_time()
        total_perf_time = global_end_perf - global_start_perf
        total_process_time = global_end_process - global_start_process

        print("\n===== 順序流程性能 =====")
        print(f"總執行時間 (wall clock): {total_perf_time:.4f}秒")
        print(f"總 CPU 時間: {total_process_time:.4f}秒")
        print(f"模式: {mode}")

        perf_data.append(
            {
                "執行模式": "順序模式",
                "節點": "全局",
                "llm_type": "openai",
                "串流模式": mode,
                "wall_clock_time": total_perf_time,
                "cpu_time": total_process_time,
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
                "時間戳": datetime.now().isoformat(),
            }
        )

        return final_state

    except Exception as e:
        logger.exception(f"順序流程執行時發生錯誤: {e}")
        print(f"\n執行過程中發生錯誤: {e}")
        return {"error": str(e)}


async def initialize_state(state: FlowState) -> FlowState:
    """初始化流程狀態"""
    logger.info("初始化並行流程狀態")
    return state


async def true_parallel_aggregator(state: FlowState) -> FlowState:
    """真正的並行聚合器，可以處理部分結果"""
    # 開始計時
    start_perf = perf_counter()
    start_process = process_time()

    logger.info("進入真正並行聚合器")

    # 初始化節點指標
    node_metrics = {"start_perf": start_perf, "start_process": start_process}

    # 檢查是否已有部分結果
    agent2_done = "agent_2_output" in state
    agent3_done = "agent_3_output" in state

    # 如果兩個代理都完成了，直接進行聚合
    if agent2_done and agent3_done:
        logger.info("兩個代理都已完成，進行聚合")
        aggregated_content = f"""來自 Agent 2 的觀點:\n{state.get('agent_2_output', '')}\n\n來自 Agent 3 的觀點:\n{state.get('agent_3_output', '')}"""

        # 記錄聚合後的內容
        logger.debug(f"聚合後內容: {aggregated_content[:100]}...")

        # 計算時間差
        end_perf = perf_counter()
        end_process = process_time()
        node_metrics["end_perf"] = end_perf
        node_metrics["end_process"] = end_process
        node_metrics["duration_perf"] = end_perf - start_perf
        node_metrics["duration_process"] = end_process - start_process

        # 添加性能指標到結果
        if "performance_metrics" not in state:
            state["performance_metrics"] = {}
        state["performance_metrics"]["true_parallel_aggregator"] = node_metrics

        return {**state, "aggregated_content": aggregated_content}

    # 如果只有部分結果，返回部分聚合結果
    logger.info(f"部分代理已完成: Agent2={agent2_done}, Agent3={agent3_done}")
    partial_content = []

    if agent2_done:
        partial_content.append(
            f"來自 Agent 2 的觀點:\n{state.get('agent_2_output', '')}"
        )

    if agent3_done:
        partial_content.append(
            f"來自 Agent 3 的觀點:\n{state.get('agent_3_output', '')}"
        )

    # 如果有部分內容，進行部分聚合
    if partial_content:
        aggregated_content = "\n\n".join(partial_content)
        logger.debug(f"部分聚合後內容: {aggregated_content[:100]}...")

        # 記錄時間
        end_perf = perf_counter()
        end_process = process_time()
        node_metrics["end_perf"] = end_perf
        node_metrics["end_process"] = end_process
        node_metrics["duration_perf"] = end_perf - start_perf
        node_metrics["duration_process"] = end_process - start_process

        # 添加性能指標到結果
        if "performance_metrics" not in state:
            state["performance_metrics"] = {}
        state["performance_metrics"]["true_parallel_aggregator"] = node_metrics

        return {**state, "aggregated_content": aggregated_content, "is_partial": True}

    # 如果沒有任何結果，返回等待
    logger.info("暫無代理完成，等待結果")
    return state


async def run_sequential_improved(initial_message: str, use_streaming: bool = True):
    """執行順序流程（非並行）"""
    # 記錄起始時間
    global_start_perf = perf_counter()
    global_start_process = process_time()

    mode = "串流" if use_streaming else "非串流"
    logger.info(f"開始執行【順序流程】，使用{mode}模式")
    print(f"\n===== 開始執行【順序流程】({mode}) =====")

    workflow = StateGraph(FlowState)
    from functools import partial

    llm2 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
    )
    llm3 = ChatOpenAI(
        model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
    )

    llm_agent_2_node_partial = partial(llm_agent_node, agent_id=2, llm_runnable=llm2)
    llm_agent_3_node_partial = partial(llm_agent_node, agent_id=3, llm_runnable=llm3)

    workflow.add_node("start", lambda x: x)
    workflow.add_node("llm_agent_2", llm_agent_2_node_partial)
    workflow.add_node("llm_agent_3", llm_agent_3_node_partial)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("final_llm", final_llm_node)
    workflow.set_entry_point("start")
    workflow.add_edge("start", "llm_agent_2")
    workflow.add_edge("llm_agent_2", "llm_agent_3")
    workflow.add_edge("llm_agent_3", "aggregator")
    workflow.add_edge("aggregator", "final_llm")
    workflow.add_edge("final_llm", END)
    app = workflow.compile()
    logger.info("順序執行流程圖構建完成 (OpenAI LLM only)")

    initial_state: FlowState = {
        "source_input": initial_message,
        "use_streaming": mode,
        "執行模式": "順序模式",
        "llm_type": "openai",
    }
    logger.info(f"初始查詢: {initial_message}")

    # 串流模式特殊處理
    first_token_time = None

    if use_streaming:
        # 建立自訂 callback 處理器
        class FirstTokenCallback:
            def __init__(self):
                self.first_token_received = False

            async def on_llm_new_token(self, token, **kwargs):
                nonlocal first_token_time
                if not self.first_token_received:
                    first_token_time = perf_counter()
                    self.first_token_received = True
                    # 立即計算並記錄時間到第一個 token
                    ttft = first_token_time - global_start_perf
                    print(f"\n--- 第一個 token 響應時間: {ttft:.4f}秒 ---")
                    # 記錄到性能數據
                    perf_data.append(
                        {
                            "執行模式": "順序執行",
                            "節點": "首次響應",
                            "llm_type": "openai",
                            "串流模式": "串流",
                            "wall_clock_time": ttft,
                            "time_to_first_token": ttft,
                            "時間戳": datetime.now().isoformat(),
                        }
                    )

        # 為 LLM 添加 callback
        callbacks = [FirstTokenCallback()]
        llm2.callbacks = callbacks
        llm3.callbacks = callbacks

    # ... 剩餘執行代碼 ...
    try:
        final_state = await app.ainvoke(initial_state)
        logger.info("順序流程執行完畢")

        global_end_perf = perf_counter()
        global_end_process = process_time()
        total_perf_time = global_end_perf - global_start_perf
        total_process_time = global_end_process - global_start_process

        print("\n===== 順序流程性能 =====")
        print(f"總執行時間 (wall clock): {total_perf_time:.4f}秒")
        print(f"總 CPU 時間: {total_process_time:.4f}秒")
        print(f"模式: {mode}")

        perf_data.append(
            {
                "執行模式": "順序模式",
                "節點": "全局",
                "llm_type": "openai",
                "串流模式": mode,
                "wall_clock_time": total_perf_time,
                "cpu_time": total_process_time,
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
                "時間戳": datetime.now().isoformat(),
            }
        )

        return final_state

    except Exception as e:
        logger.exception(f"順序流程執行時發生錯誤: {e}")
        print(f"\n執行過程中發生錯誤: {e}")
        return {"error": str(e)}


async def run_parallel(initial_message: str, use_streaming: bool = True):
    """執行並行流程（OpenAI LLM only）"""
    global_start_perf = perf_counter()
    global_start_process = process_time()

    mode = "串流" if use_streaming else "非串流"
    logger.info(f"開始執行【並行流程】，使用{mode}模式")
    print(f"\n===== 開始執行【並行流程】({mode}) =====")

    app = build_graph(use_streaming=use_streaming)

    initial_state: FlowState = {
        "source_input": initial_message,
        "use_streaming": mode,
        "執行模式": "並行模式",
        "llm_type": "openai",
    }
    logger.info(f"初始查詢: {initial_message}")

    final_state = None
    try:
        if platform.system() == "Windows":
            logger.info("Windows 系統下使用 ainvoke 模式")
            final_state = await app.ainvoke(initial_state)
            logger.info("並行流程執行完畢")
        else:
            async for event in app.astream_events(initial_state, version="v1"):
                kind = event["event"]

                if kind == "on_chat_model_stream" and use_streaming:
                    content = event["data"]["chunk"].content
                    if content:
                        print(content, end="", flush=True)
                elif kind == "on_graph_end":
                    final_state = event["data"]["output"]
                    logger.info("並行流程執行完畢")
                    break

        global_end_perf = perf_counter()
        global_end_process = process_time()
        total_perf_time = global_end_perf - global_start_perf
        total_process_time = global_end_process - global_start_process

        print("\n===== 並行流程性能 =====")
        print(f"總執行時間 (wall clock): {total_perf_time:.4f}秒")
        print(f"總 CPU 時間: {total_process_time:.4f}秒")
        print(f"模式: {mode}")

        perf_data.append(
            {
                "執行模式": "並行模式",
                "節點": "全局",
                "llm_type": "openai",
                "串流模式": mode,
                "wall_clock_time": total_perf_time,
                "cpu_time": total_process_time,
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
                "時間戳": datetime.now().isoformat(),
            }
        )

        print("\n--- 流程結束 ---")

        if final_state and isinstance(final_state, dict):
            logger.debug(
                f"最終狀態: { {k: v for k, v in final_state.items() if k != 'final_output'} }"
            )
            if error := final_state.get("error"):
                logger.error(f"流程執行中發生錯誤: {error}")
                print(f"\n錯誤: {error}")
        else:
            logger.error("未能獲取最終狀態")

        return final_state

    except Exception as e:
        logger.exception(f"並行流程執行時發生錯誤: {e}")
        print(f"\n執行過程中發生錯誤: {e}")
        return {"error": str(e)}


async def build_true_parallel_graph(use_streaming: bool = True) -> StateGraph:
    """構建真正並行處理的流程圖"""
    workflow = StateGraph(FlowState)

    # 定義節點但不添加彙總器節點
    workflow.add_node("start", initialize_state)

    # 注意：我們在這裡不預先綁定 LLM 實例，而是在 run_true_parallel 函數中建立
    workflow.add_node(
        "agent_2", lambda state: llm_agent_node(state, agent_id=2, llm_runnable=None)
    )
    workflow.add_node(
        "agent_3", lambda state: llm_agent_node(state, agent_id=3, llm_runnable=None)
    )
    workflow.add_node("true_parallel_aggregator", true_parallel_aggregator)
    workflow.add_node("final_llm", final_llm_node)

    # 設置入口點
    workflow.set_entry_point("start")

    # 添加並行邊
    workflow.add_edge("start", "agent_2")
    workflow.add_edge("start", "agent_3")

    # 串流模式下設置真正的並行聚合器
    workflow.add_edge("agent_2", "true_parallel_aggregator")
    workflow.add_edge("agent_3", "true_parallel_aggregator")
    workflow.add_edge("true_parallel_aggregator", "final_llm")
    workflow.add_edge("final_llm", END)

    # 編譯流程圖
    graph = workflow.compile()
    logger.info("真正並行處理流程圖構建完成")

    return graph


async def run_true_parallel(initial_message: str, use_streaming: bool = True):
    """執行真正的並行流程"""
    # 全局計時開始
    global_start_perf = perf_counter()
    global_start_process = process_time()

    mode = "串流" if use_streaming else "非串流"
    logger.info(f"開始執行【真正並行流程】，使用{mode}模式")
    print(f"\n===== 開始執行【真正並行流程】({mode}) =====")

    # 建立流程圖
    # 不再需要 app 變數，直接使用內建函數
    await build_true_parallel_graph(use_streaming=use_streaming)

    # 初始化狀態
    initial_state: FlowState = {
        "source_input": initial_message,
        "use_streaming": use_streaming,
        "執行模式": "真正並行執行",
        "llm_type": "openai",
    }
    logger.info(f"初始查詢: {initial_message}")

    final_results = None
    try:
        # 使用 asyncio.create_task 啟動任務
        # 在 run_true_parallel 函數中同樣創建 LLM 實例
        llm2 = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
        )
        llm3 = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming
        )
        agent2_task = asyncio.create_task(
            llm_agent_node({**initial_state}, agent_id=2, llm_runnable=llm2)
        )
        agent3_task = asyncio.create_task(
            llm_agent_node({**initial_state}, agent_id=3, llm_runnable=llm3)
        )

        # 使用 as_completed 處理先完成的結果
        pending = {agent2_task, agent3_task}
        complete_states = {}

        # 等待任一任務完成
        while pending:
            # 使用 as_completed 獲取先完成的任務
            for completed_task in asyncio.as_completed(pending):
                result = await completed_task

                if completed_task == agent2_task:
                    logger.info("Agent 2 先完成")
                    complete_states["agent_2"] = result
                    if "agent_2_output" not in complete_states:
                        complete_states["agent_2_output"] = result.get(
                            "agent_2_output", ""
                        )
                elif completed_task == agent3_task:
                    logger.info("Agent 3 先完成")
                    complete_states["agent_3"] = result
                    if "agent_3_output" not in complete_states:
                        complete_states["agent_3_output"] = result.get(
                            "agent_3_output", ""
                        )

                # 從待處理集合中移除已完成任務
                pending.remove(completed_task)

                # 如果啟用了串流模式，則進行部分聚合
                if use_streaming and len(complete_states) > 0:
                    # 部分聚合處理
                    partial_state = await true_parallel_aggregator(complete_states)

                    # 如果包含部分結果，可以開始處理最終節點
                    if "aggregated_content" in partial_state and partial_state.get(
                        "is_partial", False
                    ):
                        logger.info("開始處理部分結果")
                        # 可以選擇在這裡進行部分結果的處理，例如顯示部分結果
                        print(
                            f"\n--- 部分結果 ---\n{partial_state['aggregated_content']}"
                        )

        # 所有任務完成後，合併結果
        combined_state = {**initial_state}
        for key, value in complete_states.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    if k not in combined_state:
                        combined_state[k] = v
            else:
                combined_state[key] = value

        # 進行最終聚合
        aggregated_state = await true_parallel_aggregator(combined_state)

        # 執行最終 LLM
        final_results = await final_llm_node(aggregated_state)

        # 全局計時結束
        global_end_perf = perf_counter()
        global_end_process = process_time()
        total_perf_time = global_end_perf - global_start_perf
        total_process_time = global_end_process - global_start_process

        print("\n===== 真正並行流程性能 =====")
        print(f"總執行時間 (wall clock): {total_perf_time:.4f}秒")
        print(f"總 CPU 時間: {total_process_time:.4f}秒")
        print(f"模式: {mode}")

        perf_data.append(
            {
                "執行模式": "真正並行執行",
                "節點": "全局",
                "llm_type": "openai",
                "串流模式": mode,
                "wall_clock_time": total_perf_time,
                "cpu_time": total_process_time,
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
                "時間戳": datetime.now().isoformat(),
            }
        )

        print("\n--- 流程結束 ---")

        if final_results and isinstance(final_results, dict):
            print(
                f"\n===== 最終輸出 =====\n{final_results.get('final_output', '無輸出')}"
            )
        else:
            print("\n無法取得最終輸出")

        return final_results

    except Exception as e:
        logger.exception(f"真正並行流程執行時發生錯誤: {e}")
        print(f"\n執行過程中發生錯誤: {e}")
        return {"error": str(e)}


# 添加這個新函數來模擬 AWS Lambda 串流模式


async def run_lambda_streaming(initial_message: str):
    """模擬 AWS Lambda 串流模式 - 立即轉出處理，不等待後續完成"""
    # 全局計時開始
    global_start_perf = perf_counter()
    global_start_process = process_time()

    # 清空 lambda_node_timings 以避免累積計時
    global lambda_node_timings
    lambda_node_timings = {}

    logger.info("開始執行【AWS Lambda 轉出串流模式】")
    print("\n===== 開始執行【AWS Lambda 轉出串流模式】=====")

    # 初始化狀態 - 傳遞全局開始時間以便計算到第一個 token 的時間
    initial_state: FlowState = {
        "source_input": initial_message,
        "use_streaming": True,  # 必須是串流模式
        "執行模式": "Lambda串流模式",
        "llm_type": "openai",
        "lambda_global_start_time": global_start_perf,
        "lambda_global_start_cpu": global_start_process,
    }
    logger.info(f"初始查詢: {initial_message}")

    try:
        # 建立 LLM 實例
        llm2 = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=True
        )
        llm3 = ChatOpenAI(
            model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=True
        )

        # 使用 asyncio.create_task 啟動兩個代理但不等待它們
        agent2_task = asyncio.create_task(
            llm_agent_node({**initial_state}, agent_id=2, llm_runnable=llm2)
        )
        agent3_task = asyncio.create_task(
            llm_agent_node({**initial_state}, agent_id=3, llm_runnable=llm3)
        )

        # 建立一個任務集合
        tasks = {agent2_task, agent3_task}

        # 模擬 Lambda 處理 - 只等待第一個代理開始返回結果後就立即轉出
        try:
            # 設置一個超時，確保不會永久等待
            first_done, _ = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
                timeout=5.0,  # 最多等待5秒就轉出
            )

            if first_done:
                first_agent = list(first_done)[0]
                result = await first_agent
                agent_id = 2 if first_agent == agent2_task else 3
                logger.info(f"Agent {agent_id} 首先完成，繼續處理")

                # 記錄第一個代理的完成時間（但不是轉出點）
                agent_done_time = perf_counter() - global_start_perf
                logger.info(f"第一個代理完成時間: {agent_done_time:.4f}秒")

                # 確保將第一個完成的代理時間添加到 lambda_node_timings
                if f"Agent_{agent_id}" not in lambda_node_timings:
                    lambda_node_timings[f"Agent_{agent_id}"] = agent_done_time
                    logger.info(
                        f"Lambda模式: Agent {agent_id} 節點執行時間: {agent_done_time:.4f}秒 (來自主流程)"
                    )

                # 繼續等待其他代理完成
                remaining_tasks = tasks - first_done
                if remaining_tasks:
                    try:
                        # 繼續等待剩餘的代理，但設置一個合理的超時
                        other_done, _ = await asyncio.wait(
                            remaining_tasks, timeout=10.0
                        )
                        if other_done:
                            other_agent = list(other_done)[0]
                            # 獲取結果但不使用，只是確保任務已完成
                            _ = await other_agent
                            other_agent_id = 3 if agent_id == 2 else 2
                            logger.info(f"Agent {other_agent_id} 也完成了")
                            # 記錄第二個代理的時間
                            other_agent_time = perf_counter() - global_start_perf
                            if f"Agent_{other_agent_id}" not in lambda_node_timings:
                                lambda_node_timings[f"Agent_{other_agent_id}"] = (
                                    other_agent_time
                                )
                                logger.info(
                                    f"Lambda模式: Agent {other_agent_id} 節點執行時間: {other_agent_time:.4f}秒 (來自主流程)"
                                )
                    except asyncio.TimeoutError:
                        logger.info("等待其他代理超時，繼續處理")

                # 在這裡不停止計時，而是繼續執行聚合和最終 LLM

            else:
                # 如果超時，繼續處理
                logger.info("等待代理超時，繼續處理")

        except asyncio.TimeoutError:
            logger.info("等待代理超時，繼續處理")

        # 執行聚合處理
        logger.info("Lambda 模式執行聚合處理")
        combined_state = {**initial_state}
        for task in tasks:
            if task.done():
                try:
                    result = task.result()
                    if isinstance(result, dict):
                        for k, v in result.items():
                            if k not in combined_state:
                                combined_state[k] = v
                except Exception as e:
                    logger.error(f"獲取任務結果時發生錯誤: {e}")

        # 執行聚合節點
        await aggregator_node(combined_state)

        # 檢查是否 Final LLM 會提前退出
        if not combined_state.get("merged_output"):
            # 如果聚合內容為空，Final LLM 會提前退出，則記錄一個模擬值
            lambda_node_timings["Final_LLM"] = 0.0  # 設置為 0 表示實際未執行
            logger.info("Final LLM 節點將提前退出，設置計時為 0.0")

        # 執行最終 LLM - 這裡會計算第一個 token 的時間
        logger.info("Lambda 模式執行最終 LLM")
        await final_llm_node(combined_state)

        # 計算 Lambda 節點總執行時間
        lambda_node_total_time = calculate_lambda_total_node_time()
        logger.info(f"Lambda 節點總執行時間: {lambda_node_total_time:.4f}秒")

        # 記錄總執行時間
        total_exec_time = perf_counter() - global_start_perf
        logger.info(f"Lambda 總執行時間 (wall clock): {total_exec_time:.4f}秒")

        # 記錄到性能數據
        perf_data.append(
            {
                "執行模式": "Lambda串流模式",
                "節點": "Lambda主線程",
                "llm_type": "openai",
                "串流模式": "串流",
                "wall_clock_time": total_exec_time,
                "cpu_time": process_time() - global_start_process,
                "api_wall_clock_time": total_exec_time,
                "api_cpu_time": process_time() - global_start_process,
                "時間戳": datetime.now().isoformat(),
            }
        )

        # 記錄 Lambda 節點總執行時間
        perf_data.append(
            {
                "執行模式": "Lambda串流模式",
                "節點": "Lambda節點總和",
                "llm_type": "openai",
                "串流模式": "串流",
                "wall_clock_time": lambda_node_total_time,
                "cpu_time": process_time() - global_start_process,
                "api_wall_clock_time": lambda_node_total_time,
                "api_cpu_time": process_time() - global_start_process,
                "時間戳": datetime.now().isoformat(),
            }
        )

        # Lambda 函數已完成，但我們不在這裡停止計時
        # 計時已在 final_llm_node 中的第一個 token 時完成
        print("\n--- Lambda 模式流程完成 ---")
        print(f"Lambda 總執行時間: {total_exec_time:.6f}秒")
        print(f"Lambda 節點總執行時間: {lambda_node_total_time:.6f}秒")

        # 顯示各節點時間明細
        print("\n各節點實際執行時間明細:")
        for node, duration in lambda_node_timings.items():
            print(
                f"  {node}: {duration:.6f}秒 ({duration/lambda_node_total_time*100:.2f}%)"
            )

        # 如果有首個token響應時間記錄，計算效率比
        lambda_entries = [
            e
            for e in perf_data
            if e["執行模式"] == "Lambda轉出串流" and e["節點"] == "首個Token"
        ]
        if lambda_entries and lambda_node_total_time > 0:
            lambda_response_time = lambda_entries[0]["wall_clock_time"]
            efficiency_ratio = lambda_response_time / lambda_node_total_time
            print(f"Lambda效率比 (端到端時間/節點總時間): {efficiency_ratio:.2f}倍")
            if efficiency_ratio > 1:
                print(
                    f"約有 {(efficiency_ratio-1)*100:.1f}% 的時間用於網路傳輸和其他開銷"
                )
            else:
                print("節點執行時間總和大於端到端回應時間，可能是由於並行處理效應")

        # 確保所有 CSV 數據被正確標記
        for entry in perf_data:
            if not entry.get("執行模式") or entry["執行模式"] == "未知":
                entry["執行模式"] = "Lambda轉出串流"
            if not entry.get("串流模式") or entry["串流模式"] == "未知":
                entry["串流模式"] = "串流"

        # 返回 Lambda 函數的結果
        return {
            "status": "streaming_completed",
            "message": "Lambda 流程完成",
        }

    except Exception as e:
        logger.exception(f"Lambda 串流模式執行時發生錯誤: {e}")
        print(f"\n執行過程中發生錯誤: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    if platform.system() == "Windows":
        logger.info("在 Windows 上設置 WindowsSelectorEventLoopPolicy")
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    elif has_uvloop:
        try:
            uvloop.install()
            logger.info("使用 uvloop 加速異步操作")
        except Exception as e:
            logger.warning(f"無法安裝 uvloop: {e}")
    else:
        logger.info("使用標準事件循環")

    # 執行順序/並行、串流/非串流共 4 組
    import gc
    import time
    
    from prompt import api_request_data
    
    # 首先執行 Lambda 串流模式測試
    print("\n=== 測試: Lambda串流模式 ===")
    try:
        asyncio.run(run_lambda_streaming(api_request_data))
        # 立即保存並分析數據
        save_performance_data()
        analyze_performance_data()
    finally:
        # 強制釋放 event loop 資源，避免 Windows 下 httpx/anyio 報錯
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.5)

    # 執行其他模式測試
    for mode in ["順序", "並行"]:
        for streaming in [False, True]:
            print(f"\n=== 測試: {mode}執行, 串流={streaming}, llm=openai ===")
            try:
                if mode == "順序":
                    asyncio.run(
                        run_sequential(api_request_data, use_streaming=streaming)
                    )
                else:
                    asyncio.run(run_parallel(api_request_data, use_streaming=streaming))
            finally:
                # 強制釋放 event loop 資源，避免 Windows 下 httpx/anyio 報錯
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        loop.close()
                except Exception:
                    pass
                gc.collect()
                time.sleep(0.5)

    # 最後再次保存並分析所有數據
    save_performance_data()
    analyze_performance_data()

    print("\n===== 性能比較摘要 =====")
    print("各種執行模式下的性能數據已保存到CSV文件中")
    print("執行模式間的比較可協助評估併行處理對性能的影響")
