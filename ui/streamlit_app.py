import json

import streamlit as st

from core.agent import run_agent


def render_app() -> None:
    question = st.text_area("问题输入框", height=120)

    if st.button("提交"):
        if not question.strip():
            st.warning("请输入问题。")
            return

        with st.spinner("正在处理..."):
            try:
                result = run_agent(question.strip())
            except Exception as exc:
                st.error(f"调用 Agent 失败：{exc}")
                return

        intent = result["intent_analysis"]
        metrics = result["metrics"]

        st.markdown("---")
        st.markdown("### Intent Analysis")
        st.write(f"Need Search: {str(intent['need_search']).lower()}")
        st.markdown("Reason:")
        st.write(intent.get("reason", ""))

        st.markdown("---")
        st.markdown("### Agent Trace")
        for step in result["trace_steps"]:
            loop = step["loop"]
            st.markdown(f"**Thought {loop}**")
            st.write(step.get("thought", ""))

            if "action" in step:
                st.markdown(f"**Action {loop}**")
                st.code(
                    json.dumps(step["action"], ensure_ascii=False, indent=2),
                    language="json",
                )

            if "observation" in step:
                st.markdown(f"**Observation {loop}**")
                st.code(
                    json.dumps(step["observation"], ensure_ascii=False, indent=2),
                    language="json",
                )

            if "final_answer" in step:
                st.markdown("**Final Answer**")
                st.write(step["final_answer"])

        st.markdown("---")
        st.markdown("### Run Metrics")
        st.write(f"Token: {metrics['tokens']}")
        st.write(f"Latency: {metrics['latency']}s")
        st.write(f"Search Calls: {metrics['search_calls']}")
