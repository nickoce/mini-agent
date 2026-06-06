import json

import streamlit as st

from agent import run_agent


def main() -> None:
    st.set_page_config(page_title="Agent MVP", page_icon="Agent", layout="centered")

    question = st.text_area("问题输入框", height=120)

    if st.button("提交"):
        if not question.strip():
            st.warning("请输入问题。")
            return

        with st.spinner("Agent 正在运行..."):
            try:
                result = run_agent(question.strip())
            except Exception as exc:
                st.error(f"运行失败：{exc}")
                return

        intent = result["intent"]
        metrics = result["metrics"]

        st.markdown("---")
        st.subheader("Intent Analysis")
        st.write(f"Need Search: {str(intent['need_search']).lower()}")
        st.markdown("Reason:")
        st.write(intent.get("reason", ""))

        st.markdown("---")
        st.subheader("Agent Trace")
        for item in result["trace"]:
            st.markdown(f"**{item['type']}**")
            value = item["value"]
            if isinstance(value, str):
                st.write(value)
            else:
                st.code(json.dumps(value, ensure_ascii=False, indent=2), language="json")

        st.markdown("---")
        st.subheader("Final Answer")
        st.write(result["final_answer"])

        st.markdown("---")
        st.subheader("Run Metrics")
        st.write(f"Token: {metrics['tokens']}")
        st.write(f"Latency: {metrics['latency']}s")
        st.write(f"Search Calls: {metrics['search_calls']}")


if __name__ == "__main__":
    main()
