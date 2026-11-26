import streamlit as st
from openai import OpenAI
import time

# ========== 火山方舟配置（云端部署专用，从st.secrets读取） ==========
ARK_API_KEY = st.secrets["ARK_API_KEY"]
MODEL_ID = st.secrets["MODEL_ID"]
BASE_URL = st.secrets["BASE_URL"]
# =====================================================================================

st.set_page_config(page_title="启发式AI学习助手｜CTB完整版", layout="wide")

# 移动端适配（放在set_page_config之后）
st.markdown("""
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0">
""", unsafe_allow_html=True)

# ===================== 全局初始化 =====================
def global_init():
    if "cfg" not in st.session_state:
        st.session_state.cfg = {
            "grade": "高一",
            "subject": "数学",
            "intensity": "标准",
            "curriculum": "国际课程",
            "cool_until": 0,       # 请求冷却时间戳
            "思辨追问开关": True
        }
    # 题目引导会话（修正key名称，和render函数对齐）
    if "q_session" not in st.session_state:
        st.session_state.q_session = {
            "question": "",
            "student_answer": "",
            "history": [],
            "level": 0,    # 0待输入，1‑4提示，5复盘
            "last_ai": "",
            "ask_answer_cnt": 0,
            "level_history": [],
            "replay_text": "",
            "replay_note_id": None
        }
    # 知识点梳理会话
    if "k_session" not in st.session_state:
        st.session_state.k_session = {
            "input_text": "",
            "mode": "提纲生成", # 提纲生成 / 概念辨析 / 单元专题框架
            "ai_result": "",
            "last_source": ""
        }
    # 笔记系统：每条笔记带版本快照、来源标记
    if "note_store" not in st.session_state:
        st.session_state.note_store = []
    # 埋点内存日志（演示用，重启丢失）
    if "event_log" not in st.session_state:
        st.session_state.event_log = []

global_init()
cfg = st.session_state.cfg
client = OpenAI(api_key=ARK_API_KEY, base_url=BASE_URL)

# ===================== 全局工具函数 =====================
def add_event(evt_type, payload:dict):
    """埋点日志记录"""
    log_item = {
        "ts": time.time(),
        "event": evt_type,
        "grade": cfg["grade"],
        "subject": cfg["subject"],
        **payload
    }
    st.session_state.event_log.append(log_item)

def is_cooling() -> bool:
    now = time.time()
    return now < cfg["cool_until"]

def set_cool(sec=60):
    cfg["cool_until"] = time.time() + sec

def stream_ai_call(sys_prompt:str, user_prompt:str, temperature=0.4):
    """统一流式大模型调用，全局异常捕获"""
    if is_cooling():
        return None, "请求过于频繁，请稍后再试"
    try:
        stream = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role":"system","content":sys_prompt},{"role":"user","content":user_prompt}],
            temperature=temperature,
            stream=True
        )
        full = ""
        box = st.empty()
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                full += chunk.choices[0].delta.content
                box.markdown(full)
        return full, None
    except Exception as e:
        return None, f"接口异常：{str(e)}"

def detect_ask_answer(text:str) -> bool:
    """简单关键词检测索要答案"""
    kw = ["直接告诉我答案","给我答案","算出来","求结果","告诉我选什么","直接解这道题"]
    return any(k in text for k in kw)

def detect_full_note_request(text:str) -> bool:
    """检测要求AI直接写完整笔记"""
    kw = ["帮我写完整笔记","直接写笔记","生成完整笔记","全部写好"]
    return any(k in text for k in kw)

# ===================== 题目引导模块【已修改：不限次数，持续多轮启发，严禁输出答案】 =====================
def render_question_module():
    st.subheader("📖 启发式题目")
    qs = st.session_state["q_session"]

    qs["question"] = st.text_area("已知的题目", value=qs["question"], height=120, disabled=False)
    student_input = st.text_area("写下你的想法：想到了什么，在哪？", value=qs["student_answer"], height=140)
    qs["student_answer"] = student_input

    col1, col2, col3 = st.columns([1,1,1])
    with col1:
        btn_next = st.button("给我下一层提示")
    with col2:
        btn_reset = st.button("重置本会话")
    with col3:
        btn_review = st.button("进入复盘(第5级)")

    if btn_reset:
        qs["level"] = 0
        qs["history"] = []
        qs["student_answer"] = ""
        st.rerun()

    if btn_review:
        qs["level"] = 5
        st.rerun()

    if btn_next and student_input.strip() != "":
        # 构建对话历史，带上全部上下文
        chat_history = ""
        for item in qs["history"]:
            chat_history += f"学生：{item['student']}\n引导：{item['ai']}\n"

        sys_prompt = """
你是一名启发式辅导助手，严格遵守下面规则：
1. 绝对不能直接算出题目的最终答案，不给出数值结果，不完整解出题目；
2. 不替学生计算、不列出完整解方程步骤；
3. 只使用提问、点拨、指出思维漏洞、引导分类的方式启发学生自己思考；
4. 根据学生当前给出的思路，针对性追问，引导他发现自己哪里考虑不周；
5. 输出风格简短，贴合高中十年级学生，不要长篇大论；
6. 如果学生思路跑偏，温和指出方向问题，继续引导，不要直接纠正到底；
7. 禁止出现“答案是”“解得”“x=”“a=”这类字样。
"""
        user_prompt = f"""
题目：{qs["question"]}
历史对话：
{chat_history}
学生最新想法：{student_input}
请输出一轮启发引导：
"""
        # 调用LLM
        resp = client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role":"system", "content": sys_prompt},
                {"role":"user", "content": user_prompt}
            ],
            temperature=0.3
        )
        ai_reply = resp.choices[0].message.content

        qs["history"].append({
            "student": student_input,
            "ai": ai_reply
        })
        qs["level"] += 1
        st.rerun()

    # 渲染历史对话
    st.divider()
    st.markdown("**AI启发提示：**")
    if len(qs["history"]) > 0:
        last = qs["history"][-1]
        st.write(last["ai"])
        st.info("💡 你可以继续在上方输入框补充你的新思考，再次点击【给我下一层提示】继续提问")

# ===================== 知识点梳理模块（三种模式） =====================
def render_knowledge_module():
    ks = st.session_state.k_session
    st.subheader("🧩 知识点整理梳理")
    ks["input_text"] = st.text_input("输入知识点/专题/需要辨析的概念", value=ks["input_text"])
    mode_sel = st.radio("工作模式", ["提纲生成","概念辨析","单元专题框架"], horizontal=True)
    ks["mode"] = mode_sel
    btn_gen = st.button("生成知识框架素材", disabled=len(ks["input_text"].strip())<3 or is_cooling())

    if btn_gen:
        if detect_full_note_request(ks["input_text"]):
            add_event("detect_fullnote", {})
            st.error("我不能直接为你生成完整可抄写的笔记，请输入知识点主题，由我输出空白框架，你自行填充。")
            set_cool(20)
            st.rerun()
        mode_instr = {
            "提纲生成":"只输出结构化空白提纲要点，不要填充完整内容，留给学生填写，末尾附带思辨思考题。",
            "概念辨析":"针对输入的多个概念做双向对比：定义、适用条件、易混淆点、反例；禁止单向长篇讲解，不输出完整成品笔记。",
            "单元专题框架":"输出单元专题层级空白模块框架，只留标题空位，不填充细节内容，末尾附带思辨思考题。"
        }
        sys_p = f"""你是学习助手，面向{cfg['curriculum']} {cfg['grade']} {cfg['subject']}学生。
硬性约束：禁止生成完整可直接抄写的成品笔记，只输出框架/提纲素材，学生自己填充内容。
{mode_instr[ks['mode']]}
"""
        usr_p = f"用户输入：{ks['input_text']}"
        res, err = stream_ai_call(sys_p, usr_p)
        if err:
            st.error(err)
        else:
            ks["ai_result"] = res
            add_event("knowledge_gen", {"mode":ks["mode"]})
        st.rerun()

    if ks["ai_result"]:
        st.divider()
        st.markdown(ks["ai_result"])
        if st.button("把本框架素材导入笔记"):
            note_id = len(st.session_state.note_store)
            new_note = {
                "note_id": note_id,
                "title": f"知识点框架_{note_id}",
                "content": f"【知识点框架素材】\n{ks['ai_result']}",
                "source":"knowledge_import",
                "version_list": [{"content":f"【知识点框架素材】\n{ks['ai_result']}"}]
            }
            st.session_state.note_store.append(new_note)
            add_event("k_import_note", {"note_id":note_id})
            st.success(f"素材已新建笔记 ID={note_id}")

# ===================== 在线笔记模块（简易版本快照、润色拦截） =====================
def render_note_module():
    st.subheader("📝 在线笔记")
    note_list = st.session_state.note_store
    opt = ["新建笔记"] + [f'[{n["note_id"]}] {n["title"]}' for n in note_list]
    sel = st.selectbox("选择笔记", opt)
    current_note = None
    if sel == "新建笔记":
        nt = st.text_input("笔记标题")
        nc = st.text_area("笔记正文内容", height=260)
        btn_save_new = st.button("保存新建笔记", disabled=len(nt.strip())<1)
        if btn_save_new:
            nid = len(note_list)
            obj = {
                "note_id":nid,
                "title":nt,
                "content":nc,
                "source":"manual_create",
                "version_list":[{"content":nc}]
            }
            note_list.append(obj)
            add_event("note_new", {"note_id":nid})
            st.success(f"已保存，笔记ID {nid}")
    else:
        idx = int(sel.split("]")[0].strip("["))
        for item in note_list:
            if item["note_id"] == idx:
                current_note = item
                break
        if current_note:
            current_note["title"] = st.text_input("笔记标题", value=current_note["title"])
            current_note["content"] = st.text_area("笔记正文（编辑修改）", value=current_note["content"], height=300)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                if st.button("保存修改 & 创建版本快照"):
                    current_note["version_list"].append({"content":current_note["content"]})
                    add_event("note_save_edit", {"note_id":current_note["note_id"]})
                    st.success("已保存，生成新版本快照")
            with col_b:
                btn_polish = st.button("AI辅助润色（仅优化语句，不重写）", disabled=len(current_note["content"].strip())<10 or is_cooling())
            with col_c:
                ver_idx = st.selectbox("恢复历史版本", options=list(range(len(current_note["version_list"]))))
                if st.button("执行恢复"):
                    sel_ver = current_note["version_list"][ver_idx]
                    current_note["content"] = sel_ver["content"]
                    st.rerun()
            if btn_polish:
                sys_p = """你是笔记润色助手。只优化语句通顺度、排版逻辑。严禁大面积重写、不能替换学生原有核心想法，不能替学生写新内容。润色结束输出1‑2个思辨思考题。"""
                usr_p = f"笔记原文：{current_note['content']}"
                out, err = stream_ai_call(sys_p, usr_p)
                if err:
                    st.error(err)
                else:
                    st.markdown("### 润色预览（需要手动确认保存才会写入笔记）")
                    st.markdown(out)
                    add_event("note_polish", {"note_id":current_note["note_id"]})

# ===================== 侧边全局设置 =====================
with st.sidebar:
    st.header("全局设置")
    cfg["curriculum"] = st.selectbox("课程体系", ["国内课程","国际课程"])
    cfg["grade"] = st.selectbox("年级", ["初一","初二","初三","高一","高二","高三"])
    cfg["subject"] = st.selectbox("学科", ["数学","物理","化学","经济","语文","英语"])
    cfg["intensity"] = st.selectbox("启发强度", ["少","标准","多"])
    cfg["思辨追问开关"] = st.checkbox("开启思辨追问", value=cfg["思辨追问开关"])
    st.divider()
    st.caption("⚠️ 内存演示：刷新页面全部数据丢失")

# ===================== 主页面布局 + 顶部常驻警示 =====================
st.markdown("""
<div style="background:#fff2cc; padding:10px; border-radius:6px; margin-bottom:15px;">
<b>⚠️ AI生成可能有误，注意辨别；AI仅为启发，请务必独立思考，核对内容正误</b>
</div>
""", unsafe_allow_html=True)

tab_q, tab_k, tab_n = st.tabs(["📖题目思路引导","🧩知识点梳理","📝在线笔记"])
with tab_q:
    render_question_module()
with tab_k:
    render_knowledge_module()
with tab_n:
    render_note_module()





