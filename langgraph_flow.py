import asyncio
import csv
import os
from datetime import datetime
from pathlib import Path
from time import perf_counter, process_time
from typing import Any, Literal, NotRequired, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger

logger.add("langgraph_flow.log", rotation="10 MB", retention="7 days", level="DEBUG")
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    logger.warning("OPENAI_API_KEY 環境變數未設置，OpenAI LLM 可能無法工作。")
    raise ValueError("OPENAI_API_KEY environment variable not set.")

class FlowState(TypedDict):
    source_input: NotRequired[str]
    llm2_output: NotRequired[str]
    llm3_output: NotRequired[str]
    merged_output: NotRequired[str]
    final_output: NotRequired[str]
    lambda_global_start_perf: NotRequired[float]
    lambda_first_token_perf: NotRequired[float]

perf_data = []
lambda_node_timings = {}
output_dir = Path("performance_outputs")
output_dir.mkdir(exist_ok=True)

CSV_HEADER = [
    "執行模式", "節點", "llm_type", "串流模式", "wall_clock_time", "cpu_time",
    "api_wall_clock_time", "api_cpu_time", "input", "output", "時間戳", "error", "節點佔比"
]

def get_csv_row(row: dict) -> dict:
    return {
        "執行模式": row.get("mode") or row.get("執行模式") or "",
        "節點": row.get("node") or row.get("節點") or "",
        "llm_type": row.get("llm_type") or "",
        "串流模式": row.get("串流模式") or "",
        "wall_clock_time": row.get("wall_clock_time") or "",
        "cpu_time": row.get("cpu_time") or "",
        "api_wall_clock_time": row.get("api_wall_clock_time") or "",
        "api_cpu_time": row.get("api_cpu_time") or "",
        "input": row.get("input") or "",
        "output": row.get("output") or "",
        "時間戳": row.get("時間戳") or "",
        "error": row.get("error") or "",
        "節點佔比": row.get("節點佔比") or "",
    }

async def llm_agent_node(state: FlowState, agent_id: int, llm_runnable, system_prompt=None) -> dict[str, Any]:
    logger.info(f"進入 LLM Agent {agent_id} 節點 (OpenAI)")
    message = state["source_input"]
    is_lambda = state.get("mode") == "lambda"
    start_perf = perf_counter()
    start_process = process_time()
    if not message:
        logger.warning(f"LLM Agent {agent_id}: 原始訊息為空")
        return {f"llm{agent_id}_output": "error: empty input", f"llm{agent_id}_start_perf": start_perf}
    if system_prompt is None:
        system_prompt = "你是專業客服 AI，請根據用戶需求給予簡潔、禮貌且專業的回應。"
    system_message = SystemMessage(content=system_prompt)
    human_message = HumanMessage(content=message)
    content = ""
    duration_to_first_token = None
    try:
        if is_lambda and hasattr(llm_runnable, "astream"):
            first_token = False
            async for chunk in llm_runnable.astream([system_message, human_message]):
                if chunk.content and not first_token:
                    duration_to_first_token = perf_counter() - start_perf
                    lambda_node_timings[f"Agent_{agent_id}"] = duration_to_first_token
                    first_token = True
                if chunk.content:
                    content += chunk.content
            if not first_token:
                duration_to_first_token = perf_counter() - start_perf
                lambda_node_timings[f"Agent_{agent_id}"] = duration_to_first_token
        else:
            response = await llm_runnable.ainvoke([system_message, human_message])
            content = response.content if hasattr(response, "content") else str(response)
            if is_lambda:
                lambda_node_timings[f"Agent_{agent_id}"] = perf_counter() - start_perf
    except Exception as e:
        logger.exception(f"LLM Agent {agent_id} 執行錯誤: {e}")
        content = f"error: {str(e)}"
        if is_lambda and duration_to_first_token is None:
            lambda_node_timings[f"Agent_{agent_id}"] = perf_counter() - start_perf
    end_perf = perf_counter()
    end_process = process_time()
    perf_data.append({
        "mode": state.get("mode", ""),
        "node": f"Agent_{agent_id}",
        "llm_type": "openai",
        "串流模式": state.get("串流模式", ""),
        "wall_clock_time": end_perf - start_perf,
        "cpu_time": end_process - start_process,
        "input": message,
        "output": content,
        "時間戳": datetime.now().isoformat(),
        "error": "" if "error" not in content else content,
        "節點佔比": "",
    })
    return {f"llm{agent_id}_output": content, f"llm{agent_id}_start_perf": start_perf}

async def aggregator_node(state: FlowState) -> dict[str, Any]:
    logger.info("進入 Aggregator 節點")
    llm2 = state.get("llm2_output", "")
    llm3 = state.get("llm3_output", "")
    is_lambda = state.get("mode") == "lambda"
    start_perf = perf_counter()
    start_process = process_time()
    merged_output = ""
    duration_to_first = None
    if not llm2.strip() and not llm3.strip():
        merged_output = "兩個 Agent 均未返回有效回應。"
    else:
        merged_output = f"來自 Agent 2 的觀點:\n{llm2}\n\n來自 Agent 3 的觀點:\n{llm3}"
    if is_lambda:
        duration_to_first = perf_counter() - start_perf
        lambda_node_timings["Aggregator"] = duration_to_first
    end_perf = perf_counter()
    end_process = process_time()
    perf_data.append({
        "mode": state.get("mode", ""),
        "node": "Aggregator",
        "llm_type": "openai",
        "串流模式": state.get("串流模式", ""),
        "wall_clock_time": end_perf - start_perf,
        "cpu_time": end_process - start_process,
        "input": {"llm2_output": llm2, "llm3_output": llm3},
        "output": merged_output,
        "時間戳": datetime.now().isoformat(),
        "error": "",
        "節點佔比": "",
    })
    return {"merged_output": merged_output}

async def final_llm_node(state: FlowState) -> dict[Literal["final_output"], str]:
    logger.info("進入 Final LLM 節點")
    merged_output = state.get("merged_output", "")
    is_lambda = state.get("mode") == "lambda"
    use_streaming = state.get("use_streaming", True)
    start_perf = perf_counter()
    start_process = process_time()
    if not merged_output:
        return {"final_output": "聚合內容為空，無法生成最終回應。"}
    from prompt import basic_evaluator_prompt
    llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=True)
    system_message = SystemMessage(content=basic_evaluator_prompt)
    human_message = HumanMessage(content=merged_output)
    final_output = ""
    duration_to_first_token = None
    first_token = False
    try:
        if (is_lambda or use_streaming) and hasattr(llm, "astream"):
            async for chunk in llm.astream([system_message, human_message]):
                if chunk.content and not first_token:
                    duration_to_first_token = perf_counter() - start_perf
                    first_token = True
                    if is_lambda:
                        lambda_node_timings["Final_LLM"] = duration_to_first_token
                        if "lambda_global_start_perf" in state:
                            lambda_first_token_perf = perf_counter()
                            state["lambda_first_token_perf"] = lambda_first_token_perf
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    final_output += chunk.content
            if not first_token:
                duration_to_first_token = perf_counter() - start_perf
                if is_lambda:
                    lambda_node_timings["Final_LLM"] = duration_to_first_token
            print()
        else:
            async for chunk in llm.astream([system_message, human_message]):
                if chunk.content:
                    print(chunk.content, end="", flush=True)
                    final_output += chunk.content
            print()
            if is_lambda:
                lambda_node_timings["Final_LLM"] = perf_counter() - start_perf
    except Exception as e:
        logger.exception(f"Final LLM 執行錯誤: {e}")
        final_output = f"error: {str(e)}"
        if is_lambda and duration_to_first_token is None:
            lambda_node_timings["Final_LLM"] = perf_counter() - start_perf
    end_perf = perf_counter()
    end_process = process_time()
    perf_data.append({
        "mode": state.get("mode", ""),
        "node": "Final_LLM",
        "llm_type": "openai",
        "串流模式": state.get("串流模式", ""),
        "wall_clock_time": end_perf - start_perf,
        "cpu_time": end_process - start_process,
        "input": merged_output,
        "output": final_output,
        "time_to_first_token": duration_to_first_token,
        "時間戳": datetime.now().isoformat(),
        "error": "" if "error" not in final_output else final_output,
        "節點佔比": "",
    })
    return {"final_output": final_output}

def build_graph(use_streaming=True) -> StateGraph:
    workflow = StateGraph(FlowState)
    from functools import partial
    llm2 = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming)
    llm3 = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming)
    hotel_system_prompt = (
        "你是一個專業的酒店預訂助手。請協助客人了解酒店信息、提供建議、處理預訂，並回答相關問題。"
    )
    llm_agent_2_node_partial = partial(llm_agent_node, agent_id=2, llm_runnable=llm2, system_prompt=hotel_system_prompt)
    llm_agent_3_node_partial = partial(llm_agent_node, agent_id=3, llm_runnable=llm3, system_prompt=hotel_system_prompt)
    workflow.add_node("llm_agent_2", llm_agent_2_node_partial)
    workflow.add_node("llm_agent_3", llm_agent_3_node_partial)
    workflow.add_node("aggregator", aggregator_node)
    workflow.add_node("final_llm", final_llm_node)
    workflow.add_node("start", lambda x: x)
    workflow.set_entry_point("start")
    workflow.add_edge("start", "llm_agent_2")
    workflow.add_edge("start", "llm_agent_3")
    workflow.add_edge("llm_agent_3", "aggregator")
    workflow.add_edge("aggregator", "final_llm")
    workflow.add_edge("final_llm", END)
    return workflow.compile()

async def run_sequential(initial_message: str, use_streaming: bool = True):
    logger.info(f"開始執行【順序流程】，串流={use_streaming}")
    from functools import partial
    llm2 = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming)
    llm3 = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=use_streaming)
    hotel_system_prompt = (
        "你是一個專業的酒店預訂助手。請協助客人了解酒店信息、提供建議、處理預訂，並回答相關問題。"
    )
    state: FlowState = {"source_input": initial_message, "use_streaming": use_streaming, "mode": "sequential"}
    global_start_perf = perf_counter()
    global_start_process = process_time()
    state.update(await llm_agent_node(state, 2, llm2, hotel_system_prompt))
    state.update(await llm_agent_node(state, 3, llm3, hotel_system_prompt))
    state.update(await aggregator_node(state))
    state.update(await final_llm_node(state))
    global_end_perf = perf_counter()
    global_end_process = process_time()
    perf_data.append({
        "mode": "順序模式",
        "node": "全局",
        "llm_type": "openai",
        "串流模式": "串流" if use_streaming else "非串流",
        "wall_clock_time": global_end_perf - global_start_perf,
        "cpu_time": global_end_process - global_start_process,
        "api_wall_clock_time": "N/A",
        "api_cpu_time": "N/A",
        "input": "",
        "output": "",
        "時間戳": datetime.now().isoformat(),
        "error": "",
        "節點佔比": "",
    })
    return state

async def run_parallel(initial_message: str, use_streaming: bool = True):
    logger.info(f"開始執行【並行流程】，串流={use_streaming}")
    app = build_graph(use_streaming=use_streaming)
    initial_state: FlowState = {
        "source_input": initial_message,
        "use_streaming": use_streaming,
        "mode": "parallel"
    }
    global_start_perf = perf_counter()
    global_start_process = process_time()
    final_state = await app.ainvoke(initial_state)
    global_end_perf = perf_counter()
    global_end_process = process_time()
    perf_data.append({
        "mode": "並行模式",
        "node": "全局",
        "llm_type": "openai",
        "串流模式": "串流" if use_streaming else "非串流",
        "wall_clock_time": global_end_perf - global_start_perf,
        "cpu_time": global_end_process - global_start_process,
        "api_wall_clock_time": "N/A",
        "api_cpu_time": "N/A",
        "input": "",
        "output": "",
        "時間戳": datetime.now().isoformat(),
        "error": "",
        "節點佔比": "",
    })
    return final_state

async def run_lambda_streaming(initial_message: str):
    logger.info("開始執行【Lambda 串流流程】")
    from functools import partial
    llm2 = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=True)
    llm3 = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0.1, streaming=True)
    hotel_system_prompt = (
        "你是一個專業的酒店預訂助手。請協助客人了解酒店信息、提供建議、處理預訂，並回答相關問題。"
    )
    # 啟動 llm_agent_2/3，誰先完成先寫入 state，但 aggregator 必須等兩個都完成
    state: FlowState = {
        "source_input": initial_message,
        "use_streaming": True,
        "mode": "lambda"
    }
    task2 = asyncio.create_task(llm_agent_node(state, 2, llm2, hotel_system_prompt))
    task3 = asyncio.create_task(llm_agent_node(state, 3, llm3, hotel_system_prompt))
    done, pending = await asyncio.wait([task2, task3], return_when=asyncio.FIRST_COMPLETED)
    # 先將已完成的 agent 結果寫入 state
    for t in done:
        state.update(await t)
    # 等剩下的 agent 也完成
    for t in pending:
        state.update(await t)
    # 取兩 agent 最早 start_perf 作為 lambda_global_start_perf
    agent2_start = task2.result().get("llm2_start_perf") if task2.done() else None
    agent3_start = task3.result().get("llm3_start_perf") if task3.done() else None
    if agent2_start is not None and agent3_start is not None:
        lambda_global_start_perf = min(agent2_start, agent3_start)
    else:
        lambda_global_start_perf = agent2_start or agent3_start or perf_counter()
    state["lambda_global_start_perf"] = lambda_global_start_perf
    # aggregator 必須等兩個 agent 都完成
    state.update(await aggregator_node(state))
    # final_llm timing
    state.update(await final_llm_node(state))
    lambda_first_token_perf = state.get("lambda_first_token_perf")
    lambda_total_time = None
    if lambda_first_token_perf and lambda_global_start_perf:
        lambda_total_time = lambda_first_token_perf - lambda_global_start_perf
    else:
        lambda_total_time = None
    lambda_node_total_time = sum(lambda_node_timings.values()) if lambda_node_timings else 0.0
    perf_data.append({
        "mode": "Lambda串流模式",
        "node": "全局",
        "llm_type": "openai",
        "串流模式": "串流",
        "wall_clock_time": lambda_total_time if lambda_total_time is not None else 0.0,
        "cpu_time": "N/A",
        "api_wall_clock_time": lambda_total_time if lambda_total_time is not None else 0.0,
        "api_cpu_time": "N/A",
        "input": "",
        "output": "",
        "時間戳": datetime.now().isoformat(),
        "error": "",
        "節點佔比": "",
    })
    perf_data.append({
        "mode": "Lambda串流模式",
        "node": "首個Token",
        "llm_type": "openai",
        "串流模式": "串流",
        "wall_clock_time": lambda_node_timings.get("Final_LLM", 0.0),
        "cpu_time": "N/A",
        "api_wall_clock_time": lambda_node_timings.get("Final_LLM", 0.0),
        "api_cpu_time": "N/A",
        "input": "",
        "output": "",
        "時間戳": datetime.now().isoformat(),
        "error": "",
        "節點佔比": "",
    })
    perf_data.append({
        "mode": "Lambda串流模式",
        "node": "Lambda節點總和",
        "llm_type": "openai",
        "串流模式": "串流",
        "wall_clock_time": lambda_node_total_time,
        "cpu_time": "N/A",
        "api_wall_clock_time": lambda_node_total_time,
        "api_cpu_time": "N/A",
        "input": "",
        "output": "",
        "時間戳": datetime.now().isoformat(),
        "error": "",
        "節點佔比": "",
    })
    return state

def save_performance_data() -> None:
    if not perf_data:
        logger.warning("沒有性能數據可儲存")
        return
    csv_file = output_dir / f"parallel_vs_sequential_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_HEADER, extrasaction="ignore")
        writer.writeheader()
        for row in perf_data:
            writer.writerow(get_csv_row(row))
    logger.info(f"性能數據已儲存到: {csv_file}")
    print(f"\n性能數據已儲存到: {csv_file}")

def analyze_performance_data():
    if not perf_data:
        print("沒有性能數據可分析")
        return

    print("\n===== 各種模式執行時間對比 =====")
    print(f"{'執行模式':<16} | {'llm_type':<8} | {'串流模式':<8} | {'總執行時間(秒)':<15}")
    print("-" * 65)
    summary = []
    for row in perf_data:
        if row.get("node") == "全局":
            summary.append(row)
    for row in summary:
        print(f"{row.get('mode', ''):<16} | {row.get('llm_type',''):<8} | {row.get('串流模式',''):<8} | {row.get('wall_clock_time',0):<15.4f}")

    lambda_main = [r for r in perf_data if r.get("mode") == "Lambda串流模式" and r.get("node") == "全局"]
    lambda_token = [r for r in perf_data if r.get("mode") == "Lambda串流模式" and r.get("node") == "首個Token"]
    lambda_total = [r for r in perf_data if r.get("mode") == "Lambda串流模式" and r.get("node") == "Lambda節點總和"]
    print("\n===== Lambda模式效率分析 =====")
    if lambda_token:
        print(f"Lambda首個Token響應時間: {lambda_token[0]['wall_clock_time']:.4f}秒")
    if lambda_total:
        print(f"Lambda節點總執行時間: {lambda_total[0]['wall_clock_time']:.6f}秒")
    if lambda_node_timings:
        print("\n各節點實際執行時間明細:")
        total = sum(lambda_node_timings.values())
        for node, duration in lambda_node_timings.items():
            percent = (duration / total * 100) if total > 0 else 0
            print(f"  {node}: {duration:.6f}秒 ({percent:.2f}%)")
    if lambda_token and lambda_total and lambda_total[0]['wall_clock_time'] > 0:
        ratio = lambda_token[0]['wall_clock_time'] / lambda_total[0]['wall_clock_time']
        print(f"\nLambda端到端時間與節點執行時間比例: {ratio:.2f}倍")
        if ratio > 1:
            print(f"這表示約有 {(ratio-1)*100:.1f}% 的時間用於網路傳輸和其他開銷")
        else:
            print("節點執行時間總和大於端到端回應時間，可能是由於並行處理效應")

    if lambda_token:
        print("\n與其他執行模式比較:")
        lambda_time = lambda_token[0]['wall_clock_time']
        for row in summary:
            if row.get('mode') != "Lambda串流模式":
                avg_time = row.get('wall_clock_time', 0)
                if avg_time and lambda_time:
                    speedup = avg_time / lambda_time
                    print(f"  比{row.get('mode','')}快 {speedup:.2f}倍")

    print("\n===== 性能比較摘要 =====")
    print("各種執行模式下的性能數據已保存到CSV文件中")
    print("執行模式間的比較可協助評估併行處理對性能的影響")

if __name__ == "__main__":
    import time
    import gc
    from prompt import api_request_data

    print("\n=== 測試: Lambda 串流流程 ===")
    try:
        asyncio.run(run_lambda_streaming(api_request_data))
    finally:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.5)

    print("\n=== 測試: 順序流程（非串流） ===")
    try:
        asyncio.run(run_sequential(api_request_data, use_streaming=False))
    finally:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.5)

    print("\n=== 測試: 順序流程（串流） ===")
    try:
        asyncio.run(run_sequential(api_request_data, use_streaming=True))
    finally:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.5)

    print("\n=== 測試: 並行流程（非串流） ===")
    try:
        asyncio.run(run_parallel(api_request_data, use_streaming=False))
    finally:
        try:
            loop = asyncio.get_event_loop()
            if not loop.is_closed():
                loop.close()
        except Exception:
            pass
        gc.collect()
        time.sleep(0.5)

    print("\n=== 測試: 並行流程（串流） ===")
    try:
        asyncio.run(run_parallel(api_request_data, use_streaming=True))
    finally:
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
