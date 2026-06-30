import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
import time
from datetime import datetime

# 1. IMPORT CÁC HÀM CẦN THIẾT TRƯỚC TIÊN
from attention_module import load_attention_model, predict_with_attention

# 2. ĐỊNH NGHĨA HÀM CACHE ĐỂ TẢI MODEL
@st.cache_resource
def get_model():
    # Đảm bảo đường dẫn file .h5 và .pkl nằm cùng thư mục với app.py
    model, tokenizer = load_attention_model("attention_extraction_model.h5", "tokenizer_final.pkl")
    return model, tokenizer

# 3. GỌI HÀM ĐỂ LẤY MODEL VÀ TOKENIZER (Chỉ chạy 1 lần khi ứng dụng khởi động)
model, tokenizer = get_model()

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarcasm & Sentiment Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding: 1.5rem 2rem; max-width: 1600px; }

.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 1.2rem 1.5rem; text-align: center;
    box-shadow: 0 2px 12px rgba(15,23,42,0.08);
    position: relative; overflow: hidden;
}
.kpi-card::before { content:''; position:absolute; top:0;left:0;right:0; height:4px; }
.kpi-blue::before  { background: linear-gradient(90deg,#1d4ed8,#3b82f6); }
.kpi-orange::before{ background: linear-gradient(90deg,#b45309,#f59e0b); }
.kpi-red::before   { background: linear-gradient(90deg,#b91c1c,#ef4444); }
.kpi-green::before { background: linear-gradient(90deg,#047857,#10b981); }
.kpi-label { color:#334155; font-size:0.8rem; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }
.kpi-value { color:#0f172a; font-size:2.3rem; font-weight:800; margin:.2rem 0; }
.kpi-delta { font-size:0.78rem; font-weight:600; }
.delta-up   { color:#059669; }
.delta-down { color:#dc2626; }
.kpi-icon   { font-size:1.6rem; margin-bottom:.2rem; }

.section-header {
    color:#0f172a; font-size:1.3rem; font-weight:800;
    padding:.5rem .9rem; border-left:6px solid #1e3a8a; background:#eef2ff;
    margin-bottom:1.1rem; display:flex; align-items:center; gap:.5rem;
    border-radius:6px; letter-spacing:.01em;
}
.section-header-danger  { border-left-color:#b91c1c !important; background:#fee2e2 !important; color:#7f1d1d !important; }
.section-header-sarcasm { border-left-color:#6d28d9 !important; background:#ede9fe !important; color:#3b0764 !important; }
.section-header-platform{ border-left-color:#b45309 !important; background:#fef3c7 !important; color:#78350f !important; }
.section-header-trend   { border-left-color:#0e7490 !important; background:#cffafe !important; color:#164e63 !important; }

.alert-critical {
    background:#fef2f2;
    border:1px solid #fca5a5; border-left:5px solid #b91c1c;
    border-radius:8px; padding:.8rem 1rem; margin-bottom:.5rem;
    color:#7f1d1d; font-size:.85rem; font-weight:500;
}
.alert-warning {
    background:#fffbeb;
    border:1px solid #fcd34d; border-left:5px solid #b45309;
    border-radius:8px; padding:.8rem 1rem; margin-bottom:.5rem;
    color:#78350f; font-size:.85rem; font-weight:500;
}
.alert-info {
    background:#eff6ff;
    border:1px solid #93c5fd; border-left:5px solid #1d4ed8;
    border-radius:8px; padding:.8rem 1rem; margin-bottom:.5rem;
    color:#1e3a8a; font-size:.85rem; font-weight:500;
}
.sarcasm-flag-card {
    background:#f5f3ff;
    border:1px solid #c4b5fd; border-left:5px solid #6d28d9;
    border-radius:8px; padding:.8rem 1rem; margin-bottom:.6rem;
}
.sarcasm-badge { background:#6d28d9; color:white; padding:3px 10px; border-radius:12px; font-size:.78rem; font-weight:800; letter-spacing:.03em; }

#MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

# ─── HELPERS ─────────────────────────────────────────────────────────────────
DANGER_KEYWORDS_DEFAULT = {
    "hàng giả": ("critical","🚨"), "hàng fake": ("critical","🚨"),
    "dị ứng":   ("warning","⚠️"),  "bóc phốt":  ("warning","⚠️"),
    "ngộ độc":  ("critical","🚨"), "lừa đảo":   ("critical","🚨"),
    "mọc mụn":  ("warning","⚠️"),
}

def detect_danger(text: str, kw_cfg: dict) -> list:
    tl = str(text).lower()
    return [(kw, lv, ic) for kw,(lv,ic) in kw_cfg.items() if kw in tl]

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:1rem 0;'>
        <div style='font-size:2rem;'>🎯</div>
        <div style='color:#f9fafb;font-weight:700;font-size:1.1rem;'>Sarcasm Dashboard</div>
        <div style='color:#6b7280;font-size:.75rem;'>AI phân tích mỉa mai & sentiment</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### ⚙️ Thiết lập cảnh báo")
    with st.expander("🔑 Từ khóa nguy cấp", expanded=True):
        kw_text = st.text_area(
            "Mỗi từ khóa 1 dòng",
            value="hàng giả\nhàng fake\ndị ứng\nbóc phốt\nngộ độc\nlừa đảo\nmọc mụn",
            height=150
        )
        keyword_cfg = {}
        for line in kw_text.strip().split("\n"):
            kw = line.strip().lower()
            if kw:
                if any(w in kw for w in ["giả","fake","độc","đảo"]):
                    keyword_cfg[kw] = ("critical","🚨")
                else:
                    keyword_cfg[kw] = ("warning","⚠️")

    with st.expander("🎯 Ngưỡng cảnh báo"):
        hi_rating_thresh = st.slider("Rating tối thiểu để cảnh báo mỉa mai", 3, 5, 4)

    st.divider()
    st.markdown("### 🎨 Bộ lọc")
    _platform_options = []
    if "df_analyzed" in st.session_state and "Nền Tảng" in st.session_state["df_analyzed"].columns:
        _platform_options = sorted(st.session_state["df_analyzed"]["Nền Tảng"].dropna().unique().tolist())
    platform_filter = st.multiselect("Nền tảng", options=_platform_options, placeholder="Tất cả")
    rating_filter   = st.slider("Rating", 1, 5, (1, 5))

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='background:linear-gradient(135deg,#1e2d4a,#0f172a);border:1px solid #1e3a5f;
     border-radius:16px;padding:1.4rem 2rem;margin-bottom:1.5rem;'>
    <div style='color:#93c5fd;font-size:.85rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;'>
        🎯 SARCASM ALERTS &amp; SENTIMENT ANALYSIS
    </div>
    <div style='color:#ffffff;font-size:2.2rem;font-weight:800;margin-top:.25rem;text-shadow:0 2px 12px rgba(96,165,250,0.25);'>
        Dashboard Phân Tích Review
    </div>
    <div style='color:#94a3b8;font-size:.83rem;margin-top:.3rem;'>
        AI phát hiện mỉa mai • Cảnh báo từ khóa nguy cấp • Phân tích sentiment tự động
    </div>
</div>""", unsafe_allow_html=True)

# ─── INPUT TABS ──────────────────────────────────────────────────────────────
tab_upload, tab_manual = st.tabs(["📂 Upload File Excel", "✍️ Nhập Review Thủ Công"])

df_analyzed = None   # dataframe sau khi AI phân tích xong

# ── Tab 1: Upload ─────────────────────────────────────────────────────────────
with tab_upload:
    col_u1, col_u2 = st.columns([2,1])
    with col_u1:
        uploaded = st.file_uploader(
            "File Excel chỉ cần cột **Review**, **Rating**, **Ngày**, **Nền Tảng** — AI sẽ tự gán nhãn",
            type=["xlsx","xls"]
        )
    with col_u2:
        st.markdown("""
        <div style='background:#1e2530;border:1px solid #2d3748;border-radius:10px;
             padding:1rem;font-size:.8rem;color:#9ca3af;'>
        <b style='color:#e2e8f0;'>📋 Cột cần có:</b><br><br>
        • <b>Review</b> – nội dung đánh giá<br>
        • <b>Rating</b> – 1 đến 5 sao<br>
        • <b>Nền Tảng</b> – Shopee, Lazada…<br>
        • <b>Ngày</b> – ngày đăng (tuỳ chọn)<br><br>
        <span style='color:#f59e0b;'>⚡ AI sẽ tự động phân tích mỉa mai & sentiment</span>
        </div>""", unsafe_allow_html=True)

    if uploaded:
        df_raw = pd.read_excel(uploaded)
        df_raw.columns = df_raw.columns.str.strip()

        # Chuẩn hoá tên cột
        col_map = {}
        for c in df_raw.columns:
            cl = c.lower()
            if any(k in cl for k in ["review","bình luận","nội dung","comment"]): col_map[c]="Review"
            elif any(k in cl for k in ["rating","sao","điểm","score"]): col_map[c]="Rating"
            elif any(k in cl for k in ["ngày","date","thời gian"]): col_map[c]="Ngày"
            elif any(k in cl for k in ["nền tảng","platform","kênh","source"]): col_map[c]="Nền Tảng"
        df_raw = df_raw.rename(columns=col_map)

        if "Review" not in df_raw.columns:
            st.error("❌ Không tìm thấy cột 'Review'. Vui lòng kiểm tra file.")
        else:
            st.success(f"✅ Đọc được **{len(df_raw)}** reviews — nhấn nút bên dưới để AI phân tích")

            if st.button("🤖 Chạy AI phân tích", type="primary", use_container_width=True):
                reviews_list = df_raw["Review"].fillna("").astype(str).tolist()

                progress = st.progress(0, text="⏳ Đang phân tích bằng model cục bộ...")
                with st.spinner("Model LSTM đang phân tích từng review..."):
                    try:
                        results = []
                        for i, review in enumerate(reviews_list):
                            res = predict_with_attention(review, tokenizer, model)
                            results.append(res)
                            progress.progress((i + 1) / len(reviews_list))

                        progress.progress(100, text="✅ Hoàn thành!")
                        time.sleep(0.5)
                        progress.empty()

                        df_raw["Mỉa Mai"]   = ["Có" if r["prediction"] == "MIA MAI" else "Không" for r in results]
                        df_raw["Sentiment"] = ["Tiêu cực" if r["prediction"] == "MIA MAI" else "Tích cực" for r in results]
                        df_raw["Lý Do AI"]  = ["Độ tin cậy: " + str(round(r["confidence"], 2)) for r in results]

                        df_raw["Từ Khóa Nguy Cấp"] = df_raw["Review"].apply(
                            lambda x: ", ".join(kw for kw,_,_ in detect_danger(x, keyword_cfg))
                        )

                        if "Rating"   not in df_raw.columns: df_raw["Rating"]   = 3
                        if "Nền Tảng" not in df_raw.columns: df_raw["Nền Tảng"] = "Không rõ"
                        if "Ngày"     not in df_raw.columns: df_raw["Ngày"]     = pd.Timestamp.today()

                        st.session_state["df_analyzed"] = df_raw
                        st.rerun()
                    except Exception as e:
                        progress.empty()
                        st.error(f"❌ Lỗi khi chạy model: {e}")

# ── Tab 2: Manual ─────────────────────────────────────────────────────────────
with tab_manual:
    if "manual_rows" not in st.session_state:
        st.session_state.manual_rows = []

    with st.form("add_review"):
        c1, c2, c3 = st.columns([3,1,1])
        with c1:
            txt = st.text_area("Nội dung review", height=90, placeholder="Nhập review...")
        with c2:
            rt  = st.selectbox("Rating ⭐", [5,4,3,2,1])
            plt = st.selectbox("Nền tảng", ["Shopee","Lazada","Tiki","TikTok Shop","Google","Facebook","Khác"])
        with c3:
            st.markdown("<br>", unsafe_allow_html=True)
            sub = st.form_submit_button("➕ Thêm & Phân tích bằng LSTM", use_container_width=True)

        if sub and txt.strip():
            with st.spinner("Model LSTM đang phân tích..."):
                try:
                    res = predict_with_attention(txt, tokenizer, model)
                    st.session_state.manual_rows.append({
                        "Review": txt, "Rating": rt, "Nền Tảng": plt,
                        "Ngày": pd.Timestamp.today(),
                        "Mỉa Mai":   "Có" if res["prediction"] == "MIA MAI" else "Không",
                        "Sentiment": "Tiêu cực" if res["prediction"] == "MIA MAI" else "Tích cực",
                        "Lý Do AI":  "Độ tin cậy: " + str(round(res["confidence"], 2)),
                        "Từ Khóa Nguy Cấp": ", ".join(kw for kw,_,_ in detect_danger(txt, keyword_cfg))
                    })
                    st.success("✅ Đã thêm và phân tích!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Lỗi khi chạy model: {e}")

    if st.session_state.manual_rows:
        df_manual = pd.DataFrame(st.session_state.manual_rows)
        st.session_state["df_analyzed"] = df_manual
        df_analyzed = df_manual
        st.info(f"📝 Có {len(df_manual)} review thủ công đã phân tích")
        if st.button("🗑️ Xóa tất cả"):
            st.session_state.manual_rows = []
            if "df_analyzed" in st.session_state:
                del st.session_state["df_analyzed"]
            st.rerun()

# Lấy df từ session nếu chưa có
if df_analyzed is None and "df_analyzed" in st.session_state:
    df_analyzed = st.session_state["df_analyzed"]

# ─── MAIN DASHBOARD ───────────────────────────────────────────────────────────
if df_analyzed is not None and len(df_analyzed) > 0:
    df = df_analyzed.copy()

    # Bộ lọc sidebar
    if platform_filter and "Nền Tảng" in df.columns:
        df = df[df["Nền Tảng"].isin(platform_filter)]
    if "Rating" in df.columns:
        df = df[(df["Rating"] >= rating_filter[0]) & (df["Rating"] <= rating_filter[1])]
    if df.empty:
        st.warning("⚠️ Không có dữ liệu khớp bộ lọc.")
        st.stop()

    # ── Metrics ───────────────────────────────────────────────────────────────
    total      = len(df)
    sarc_mask  = df["Mỉa Mai"].str.strip().str.lower() == "có"
    total_sarc = sarc_mask.sum()
    sarc_rate  = total_sarc / total * 100
    neg_mask   = df["Sentiment"].str.lower().str.contains("tiêu cực|negative", na=False)
    neg_rate   = neg_mask.sum() / total * 100
    avg_rating = df["Rating"].mean() if "Rating" in df.columns else 0
    danger_rows = [(row, detect_danger(row.get("Review",""), keyword_cfg))
                   for _, row in df.iterrows()
                   if detect_danger(row.get("Review",""), keyword_cfg)]

    st.markdown("---")
    k1,k2,k3,k4 = st.columns(4)
    for col, icon, label, val, sub_val, cls in [
        (k1,"📊","Tổng Reviews",     f"{total:,}",       "Đã phân tích", "kpi-blue"),
        (k2,"🎭","Tỷ lệ Mỉa Mai",   f"{sarc_rate:.1f}%",f"{total_sarc} reviews","kpi-orange"),
        (k3,"😤","Sentiment Tiêu Cực",f"{neg_rate:.1f}%",f"{neg_mask.sum()} reviews","kpi-red"),
        (k4,"⭐","Rating Trung Bình",f"{avg_rating:.2f}", "Thang 1–5",
         "kpi-green" if avg_rating>=3.5 else "kpi-red"),
    ]:
        with col:
            st.markdown(f"""
            <div class='kpi-card {cls}'>
                <div class='kpi-icon'>{icon}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-value'>{val}</div>
                <div class='kpi-delta'>{sub_val}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Biểu đồ hàng 1 ────────────────────────────────────────────────────────
    c1,c2,c3 = st.columns([1.2,1,1])

    with c1:
        st.markdown("<div class='section-header'>📊 Số lượng review theo rating</div>", unsafe_allow_html=True)
        rc = df["Rating"].value_counts().sort_index().reset_index()
        rc.columns = ["Rating","Count"]
        fig1 = px.bar(rc, x="Rating", y="Count",
                      color="Rating", color_discrete_sequence=["#ef4444","#f97316","#f59e0b","#84cc16","#22c55e"],
                      text="Count")
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0",showlegend=False,height=270,
                           margin=dict(l=5,r=5,t=5,b=5),
                           xaxis=dict(gridcolor="#1f2937",tickvals=[1,2,3,4,5],
                                      ticktext=["1⭐","2⭐","3⭐","4⭐","5⭐"]),
                           yaxis=dict(gridcolor="#1f2937"))
        fig1.update_traces(textposition="outside",textfont_color="#e2e8f0")
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        st.markdown("<div class='section-header'>🎭 Phân bổ mỉa mai</div>", unsafe_allow_html=True)
        sc = df["Mỉa Mai"].value_counts()
        fig2 = go.Figure(go.Pie(
            labels=sc.index, values=sc.values, hole=0.6,
            marker_colors=["#7c3aed","#1e3a5f"],
            textinfo="percent", textfont_color="#fff"
        ))
        fig2.add_annotation(text=f"<b>{sarc_rate:.0f}%</b><br>Mỉa mai",
                            x=0.5,y=0.5,showarrow=False,font=dict(color="#e2e8f0",size=14))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",font_color="#e2e8f0",height=270,
                           margin=dict(l=5,r=5,t=5,b=5),
                           legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5))
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        st.markdown("<div class='section-header'>💬 Phân bổ Sentiment</div>", unsafe_allow_html=True)
        sentc = df["Sentiment"].value_counts()
        cmap  = {"Tích cực":"#22c55e","Tiêu cực":"#ef4444","Trung tính":"#94a3b8"}
        fig3  = go.Figure(go.Bar(
            x=sentc.values, y=sentc.index, orientation="h",
            marker_color=[cmap.get(s,"#6b7280") for s in sentc.index],
            text=sentc.values, textposition="outside", textfont_color="#e2e8f0"
        ))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                           font_color="#e2e8f0",height=270,margin=dict(l=5,r=5,t=5,b=5),
                           xaxis=dict(gridcolor="#1f2937"),yaxis=dict(gridcolor="#1f2937"))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ── Alerts & Sarcasm flags ────────────────────────────────────────────────
    col_d1,col_d2 = st.columns([1,1.2])

    with col_d1:
        st.markdown("<div class='section-header section-header-danger'>🚨 Cảnh báo từ khóa nguy cấp</div>", unsafe_allow_html=True)
        if danger_rows:
            for row, kws in danger_rows[:8]:
                kw_str   = ", ".join(f"{ic} <b>{kw}</b>" for kw,_,ic in kws)
                cls      = "alert-critical" if any(lv=="critical" for _,lv,_ in kws) else "alert-warning"
                preview  = str(row.get("Review",""))[:120]
                stars    = "⭐" * int(row.get("Rating",0))
                st.markdown(f"""
                <div class='{cls}'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                        <span>{kw_str}</span><span>{stars}</span>
                    </div>
                    <div style='color:#374151;font-size:.81rem;'>"{preview}{'...' if len(str(row.get('Review','')))>120 else ''}"</div>
                    <div style='margin-top:4px;color:#6b7280;font-size:.72rem;'>
                        🏪 {row.get("Nền Tảng","N/A")} • 📅 {str(row.get("Ngày",""))[:10]}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("<div class='alert-info'>✅ Không phát hiện từ khóa nguy cấp nào.</div>", unsafe_allow_html=True)

    with col_d2:
        st.markdown(f"<div class='section-header section-header-sarcasm'>🎭 Cảnh báo châm biếm – điểm cao ≥{hi_rating_thresh}⭐ nhưng bị gắn cờ mỉa mai</div>", unsafe_allow_html=True)
        flagged = df[sarc_mask & (df["Rating"] >= hi_rating_thresh)].sort_values("Rating", ascending=False)
        if len(flagged):
            for _, row in flagged.head(6).iterrows():
                preview = str(row.get("Review",""))[:130]
                reason  = str(row.get("Lý Do AI",""))
                st.markdown(f"""
                <div class='sarcasm-flag-card'>
                    <div style='display:flex;justify-content:space-between;margin-bottom:4px;'>
                        <span style='color:#b45309;'>{"⭐"*int(row.get("Rating",0))}</span>
                        <span class='sarcasm-badge'>🎭 MỈA MAI</span>
                    </div>
                    <div style='color:#374151;font-size:.83rem;font-style:italic;'>"{preview}{'...' if len(str(row.get('Review','')))>130 else ''}"</div>
                    {f'<div style="margin-top:5px;color:#6d28d9;font-size:.75rem;font-weight:600;">💡 {reason}</div>' if reason else ''}
                    <div style='margin-top:4px;color:#6b7280;font-size:.72rem;'>
                        🏪 {row.get("Nền Tảng","N/A")} • 😤 {row.get("Sentiment","N/A")}
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background:#1e2530;border:1px solid #2d3748;border-radius:10px;
                 padding:2rem;text-align:center;color:#6b7280;'>
                ✅ Không có review điểm cao bị gắn cờ mỉa mai.
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Hàng 4: Platform chart + Trend ───────────────────────────────────────
    ce1,ce2 = st.columns(2)

    with ce1:
        st.markdown("<div class='section-header section-header-platform'>🏪 Tỷ lệ mỉa mai theo nền tảng</div>", unsafe_allow_html=True)
        if "Nền Tảng" in df.columns:
            ps = df.groupby("Nền Tảng").agg(
                Total=("Review","count"),
                Sarc=("Mỉa Mai", lambda x:(x.str.strip().str.lower()=="có").sum())
            ).reset_index()
            ps["Rate"] = ps["Sarc"]/ps["Total"]*100
            ps = ps.sort_values("Rate", ascending=True)
            fig4 = go.Figure(go.Bar(
                x=ps["Rate"], y=ps["Nền Tảng"], orientation="h",
                marker=dict(color=ps["Rate"],colorscale=[[0,"#22c55e"],[.5,"#f59e0b"],[1,"#ef4444"]]),
                text=[f"{v:.1f}%" for v in ps["Rate"]],
                textposition="outside",textfont_color="#e2e8f0"
            ))
            fig4.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0",height=270,margin=dict(l=5,r=50,t=5,b=5),
                               xaxis=dict(gridcolor="#1f2937",ticksuffix="%"),yaxis=dict(gridcolor="#1f2937"))
            st.plotly_chart(fig4, use_container_width=True)

    with ce2:
        st.markdown("<div class='section-header section-header-trend'>📈 Xu hướng mỉa mai theo thời gian</div>", unsafe_allow_html=True)
        try:
            dft = df.copy()
            dft["Ngày"] = pd.to_datetime(dft["Ngày"], errors="coerce")
            trend = dft.dropna(subset=["Ngày"]).groupby(dft["Ngày"].dt.date).agg(
                Total=("Review","count"),
                Sarc=("Mỉa Mai", lambda x:(x.str.strip().str.lower()=="có").sum())
            ).reset_index().rename(columns={"Ngày":"Date"})
            trend["Rate"] = trend["Sarc"]/trend["Total"]*100
            fig5 = go.Figure()
            fig5.add_trace(go.Scatter(
                x=trend["Date"],y=trend["Rate"],mode="lines+markers",
                line=dict(color="#7c3aed",width=2),marker=dict(color="#a855f7",size=7),
                fill="tozeroy",fillcolor="rgba(124,58,237,0.15)",name="Tỷ lệ mỉa mai %"
            ))
            fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                               font_color="#e2e8f0",height=270,margin=dict(l=5,r=5,t=5,b=5),
                               xaxis=dict(gridcolor="#1f2937"),yaxis=dict(gridcolor="#1f2937",ticksuffix="%"),
                               showlegend=False)
            st.plotly_chart(fig5, use_container_width=True)
        except Exception:
            st.info("📅 Dữ liệu ngày không đủ để vẽ xu hướng.")

    st.markdown("---")

    # ── Bảng đầy đủ ───────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>📋 Danh sách reviews đã phân tích</div>", unsafe_allow_html=True)
    cf1,cf2,cf3 = st.columns([1,1,2])
    with cf1: fs = st.selectbox("Lọc mỉa mai",["Tất cả","Có","Không"])
    with cf2: fse= st.selectbox("Lọc sentiment",["Tất cả"]+sorted(df["Sentiment"].dropna().unique().tolist()))
    with cf3: fq = st.text_input("🔍 Tìm trong review", placeholder="Nhập từ khóa...")

    dfd = df.copy()
    if fs !="Tất cả": dfd=dfd[dfd["Mỉa Mai"].str.strip().str.lower()==fs.lower()]
    if fse!="Tất cả": dfd=dfd[dfd["Sentiment"]==fse]
    if fq:            dfd=dfd[dfd["Review"].str.contains(fq,case=False,na=False)]

    show_cols = [c for c in ["Review","Rating","Mỉa Mai","Sentiment","Lý Do AI","Nền Tảng","Ngày","Từ Khóa Nguy Cấp"]
                 if c in dfd.columns]
    st.dataframe(dfd[show_cols].reset_index(drop=True), use_container_width=True, height=340,
                 column_config={
                     "Rating":  st.column_config.NumberColumn("⭐ Rating",format="%d ⭐"),
                     "Review":  st.column_config.TextColumn("💬 Review",width="large"),
                     "Lý Do AI":st.column_config.TextColumn("🤖 Lý Do AI",width="medium"),
                 })

    st.markdown("---")
    col_x1,col_x2 = st.columns([1,3])
    with col_x1:
        csv = dfd[show_cols].to_csv(index=False,encoding="utf-8-sig")
        st.download_button("📥 Xuất CSV",data=csv.encode("utf-8-sig"),
                           file_name=f"review_analyzed_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                           mime="text/csv",use_container_width=True)
    with col_x2:
        st.markdown(f"""
        <div style='padding:.6rem 1rem;background:#1e2530;border:1px solid #2d3748;
             border-radius:8px;color:#9ca3af;font-size:.8rem;'>
            Hiển thị <b style='color:#e2e8f0;'>{len(dfd)}</b>/{total} reviews •
            Mỉa mai: <b style='color:#a855f7;'>{sarc_rate:.1f}%</b> •
            Tiêu cực: <b style='color:#ef4444;'>{neg_rate:.1f}%</b> •
            Rating TB: <b style='color:#fbbf24;'>{avg_rating:.2f}⭐</b>
        </div>""", unsafe_allow_html=True)

else:
    # ── Empty state ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='text-align:center;padding:4rem 2rem;background:#1e2530;
         border:1px dashed #3d4a5c;border-radius:16px;margin:2rem 0;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>🤖</div>
        <div style='color:#e2e8f0;font-size:1.4rem;font-weight:600;margin-bottom:.5rem;'>
            Chưa có dữ liệu để phân tích
        </div>
        <div style='color:#6b7280;font-size:.92rem;max-width:420px;margin:0 auto;'>
            Upload file Excel (chỉ cần cột Review & Rating) hoặc nhập review thủ công —
            AI sẽ tự động phát hiện mỉa mai và phân tích sentiment
        </div>
        <div style='margin-top:2rem;display:flex;gap:1rem;justify-content:center;flex-wrap:wrap;'>
            <div style='background:#252d3a;border:1px solid #374151;border-radius:8px;
                 padding:1rem;text-align:left;min-width:180px;'>
                <div style='color:#60a5fa;font-size:1.4rem;'>📂</div>
                <div style='color:#e2e8f0;font-weight:600;margin-top:.4rem;'>Upload Excel</div>
                <div style='color:#6b7280;font-size:.78rem;'>Chỉ cần Review + Rating</div>
            </div>
            <div style='background:#252d3a;border:1px solid #374151;border-radius:8px;
                 padding:1rem;text-align:left;min-width:180px;'>
                <div style='color:#a855f7;font-size:1.4rem;'>✍️</div>
                <div style='color:#e2e8f0;font-weight:600;margin-top:.4rem;'>Nhập thủ công</div>
                <div style='color:#6b7280;font-size:.78rem;'>Thêm từng review & AI phân tích ngay</div>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
