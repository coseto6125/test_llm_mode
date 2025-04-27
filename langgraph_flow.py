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
if not os.getenv("OPENAI_API_KEY_QA"):
    logger.warning("OPENAI_API_KEY_QA 環境變數未設置，OpenAI LLM 可能無法工作。")
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
    if not message:
        logger.warning(f"LLM Agent {agent_id}: 原始訊息為空")
        return {f"llm_{agent_id}_output": {"error": "Original message is empty."}}

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
        response = await llm_runnable.ainvoke([system_message, human_message])
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
                "執行模式": state.get("執行模式", "未知"),
                "節點": f"Agent_{agent_id}",
                "llm_type": "openai",
                "串流模式": state.get("use_streaming", "未知"),
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
                "執行模式": state.get("執行模式", "未知"),
                "節點": f"Agent_{agent_id}",
                "llm_type": "openai",
                "串流模式": state.get("use_streaming", "未知"),
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

    # 記錄 input/output
    perf_data.append(
        {
            "執行模式": state.get("執行模式", "未知"),
            "節點": "Aggregator",
            "llm_type": "N/A",
            "串流模式": state.get("use_streaming", "未知"),
            "wall_clock_time": "N/A",
            "cpu_time": "N/A",
            "api_wall_clock_time": "N/A",
            "api_cpu_time": "N/A",
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

    llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14", temperature=0.1, streaming=True)
    system_message = SystemMessage(content=basic_evaluator_prompt)
    human_message = HumanMessage(content=merged_output)

    try:
        print("\n--- 評估結果 ---\n", flush=True)
        output_chunks = []
        async for chunk in llm.astream([system_message, human_message]):
            if chunk.content:
                print(chunk.content, end="", flush=True)
                output_chunks.append(chunk.content)
        print("\n", flush=True)
        logger.info("Final LLM 串流輸出完成")
        # 記錄 input/output
        perf_data.append(
            {
                "執行模式": state.get("執行模式", "未知"),
                "節點": "Final_LLM",
                "llm_type": "openai",
                "串流模式": state.get("use_streaming", "未知"),
                "wall_clock_time": "N/A",
                "cpu_time": "N/A",
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
                "input": merged_output,
                "output": "".join(output_chunks),
                "時間戳": datetime.now().isoformat(),
            }
        )
        return {"final_output": "done"}

    except Exception as e:
        logger.exception("Final LLM 執行時發生錯誤")
        perf_data.append(
            {
                "執行模式": state.get("執行模式", "未知"),
                "節點": "Final_LLM",
                "llm_type": "openai",
                "串流模式": state.get("use_streaming", "未知"),
                "wall_clock_time": "N/A",
                "cpu_time": "N/A",
                "api_wall_clock_time": "N/A",
                "api_cpu_time": "N/A",
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
        model="gpt-4.1-nano-2025-04-14", temperature=0.1, streaming=use_streaming
    )
    llm3 = ChatOpenAI(
        model="gpt-4.1-nano-2025-04-14", temperature=0.1, streaming=use_streaming
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
    ]

    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in perf_data:
                # 將 input/output 轉為字串避免 CSV 格式錯誤
                row = dict(row)
                if isinstance(row.get("input"), (dict, list)):
                    row["input"] = str(row["input"])
                if isinstance(row.get("output"), (dict, list)):
                    row["output"] = str(row["output"])
                writer.writerow(row)

        logger.info(f"性能數據已儲存到: {csv_file}")
        print(f"\n性能數據已儲存到: {csv_file}")
    except Exception as e:
        logger.exception(f"儲存性能數據時發生錯誤: {e}")
        print(f"儲存性能數據時發生錯誤: {e}")


def analyze_performance_data():
    """分析性能數據並顯示比較結果"""
    if not perf_data:
        print("沒有性能數據可分析")
        return

    grouped_data = {}
    for entry in perf_data:
        if entry["節點"] == "全局":
            key = (entry["執行模式"], entry.get("llm_type", ""), entry["串流模式"])
            if key not in grouped_data:
                grouped_data[key] = []
            grouped_data[key].append(entry)

    print("\n===== 各種模式執行時間對比 =====")
    print(
        f"{'執行模式':<12} | {'llm_type':<8} | {'串流模式':<8} | {'總執行時間(秒)':<15}"
    )
    print("-" * 60)

    for (exec_mode, llm_type, stream_mode), entries in grouped_data.items():
        avg_time = sum(entry["wall_clock_time"] for entry in entries) / len(entries)
        print(f"{exec_mode:<12} | {llm_type:<8} | {stream_mode:<8} | {avg_time:<15.4f}")

    # 額外的分析可依需求擴充


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
        model="gpt-4.1-nano-2025-04-14", temperature=0.1, streaming=use_streaming
    )
    llm3 = ChatOpenAI(
        model="gpt-4.1-nano-2025-04-14", temperature=0.1, streaming=use_streaming
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
        "執行模式": "順序執行",
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
                "執行模式": "順序執行",
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
        "執行模式": "並行執行",
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
                "執行模式": "並行執行",
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

    save_performance_data()
    analyze_performance_data()

    print("\n===== 性能比較摘要 =====")
    print("各種執行模式下的性能數據已保存到CSV文件中")
    print("執行模式間的比較可協助評估併行處理對性能的影響")
