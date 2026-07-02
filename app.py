"""
Ứng dụng Streamlit: Dự báo Rủi ro (PD) bằng Logistic Regression
Chuyển thể từ notebook PPĐL_RRTD.ipynb
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

# =========================================================
# 1) CẤU HÌNH TRANG (phải là lệnh Streamlit đầu tiên)
# =========================================================
st.set_page_config(
    layout="wide",
    page_title="Dự báo Rủi ro Tín dụng (PD)",
    page_icon="📊",
)

# =========================================================
# 2) HẰNG SỐ & HÀM DÙNG CHUNG
# =========================================================

# Tập biến đầu vào X (đúng theo notebook, cell B2)
FEATURE_COLS = [
    "TC1", "TC2", "TC3", "TC4", "TC5",
    "NL1", "NL2", "NL3", "NL4",
    "DK1", "DK2", "DK3", "DK4", "DK5",
    "V1", "V2", "V3", "V4", "V5", "V6",
    "TS1", "TS2", "TS3", "TS4",
]
TARGET_COL = "PD"  # biến mục tiêu: 0 = không rủi ro, 1 = có rủi ro


@st.cache_data
def load_data(file_bytes: bytes) -> pd.DataFrame:
    """Nạp dữ liệu từ bytes của file CSV, dùng chung cho toàn app."""
    import io
    df = pd.read_csv(io.BytesIO(file_bytes))
    return df


def validate_columns(df: pd.DataFrame, required_cols: list) -> list:
    """Trả về danh sách cột còn thiếu trong df so với required_cols."""
    return [c for c in required_cols if c not in df.columns]


# =========================================================
# 3) SIDEBAR — VÙNG CẤU HÌNH
# =========================================================
with st.sidebar:
    st.header("⚙️ Cấu hình & Tải dữ liệu")

    uploaded_file = st.file_uploader(
        "Tải lên file dữ liệu (.csv)",
        type=["csv"],
        help="File cần chứa các cột biến đầu vào (TC1-TC5, NL1-NL4, DK1-DK5, V1-V6, TS1-TS4) và cột mục tiêu 'PD'.",
    )

    st.subheader("Tham số mô hình AI")
    st.caption("Mô hình: Logistic Regression (phân loại nhị phân: 0 = không rủi ro, 1 = có rủi ro)")

    test_size = st.slider(
        "Tỷ lệ tập kiểm tra (test_size)",
        min_value=0.05, max_value=0.5, value=0.10, step=0.05,
        help="Tỷ lệ dữ liệu dùng để kiểm định mô hình. Mặc định theo notebook gốc: 0.10.",
    )
    random_state = st.number_input(
        "random_state",
        min_value=0, max_value=9999, value=23, step=1,
        help="Hạt giống ngẫu nhiên để đảm bảo kết quả chia tập dữ liệu tái lập được. Mặc định theo notebook gốc: 23.",
    )

    with st.expander("Tham số nâng cao của Logistic Regression"):
        C = st.slider(
            "C (độ mạnh điều chuẩn nghịch đảo)",
            min_value=0.01, max_value=10.0, value=1.0, step=0.01,
            help="Giá trị càng nhỏ, điều chuẩn (regularization) càng mạnh. Mặc định scikit-learn: 1.0.",
        )
        max_iter = st.number_input(
            "max_iter (số vòng lặp tối đa)",
            min_value=50, max_value=2000, value=100, step=50,
            help="Số vòng lặp tối đa để thuật toán hội tụ. Mặc định scikit-learn: 100.",
        )
        solver = st.selectbox(
            "solver",
            options=["lbfgs", "liblinear", "newton-cg", "sag", "saga"],
            index=0,
            help="Thuật toán tối ưu hóa dùng để huấn luyện mô hình. Mặc định scikit-learn: lbfgs.",
        )

    st.divider()
    train_clicked = st.button(
        "🚀 Huấn luyện mô hình",
        type="primary",
        use_container_width=True,
    )

# =========================================================
# 4) HEADER — VÙNG ĐỊNH HƯỚNG
# =========================================================
st.title("📊 Ứng dụng Dự báo Rủi ro Tín dụng (PD)")
st.caption(
    "Ứng dụng huấn luyện mô hình Logistic Regression để dự báo khả năng rủi ro (PD) "
    "dựa trên các nhóm biến khảo sát: TC (Trách nhiệm), NL (Năng lực), DK (Điều kiện), "
    "V (Vốn), TS (Tài sản đảm bảo). Vui lòng tải lên dữ liệu ở thanh bên trái."
)

if uploaded_file is None:
    st.info("👈 Vui lòng tải lên file dữ liệu (.csv) ở thanh bên trái để bắt đầu.")
    st.stop()

file_bytes = uploaded_file.getvalue()
try:
    df = load_data(file_bytes)
except Exception as e:
    st.error(f"Không thể đọc file dữ liệu. Lỗi: {e}")
    st.stop()

if df.empty:
    st.error("File dữ liệu rỗng, vui lòng kiểm tra lại.")
    st.stop()

missing_cols = validate_columns(df, FEATURE_COLS + [TARGET_COL])
if missing_cols:
    st.error(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(missing_cols)}")
    st.stop()

st.caption(f"📁 Đang dùng tệp: **{uploaded_file.name}**")
st.caption(f"Số dòng: {df.shape[0]} | Số cột: {df.shape[1]}")
st.divider()

# =========================================================
# 5) KHỐI HUẤN LUYỆN (chạy khi bấm nút, lưu vào session_state)
# =========================================================
if train_clicked:
    try:
        X = df[FEATURE_COLS]
        y = df[TARGET_COL]

        x_train, x_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=int(random_state)
        )

        model = LogisticRegression(
            C=C, max_iter=int(max_iter), solver=solver, random_state=int(random_state)
        )
        model.fit(x_train, y_train)

        yhat_test = model.predict(x_test)
        yproba_test = model.predict_proba(x_test)[:, 1]

        st.session_state["model"] = model
        st.session_state["feature_cols"] = FEATURE_COLS
        st.session_state["results"] = {
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
            "yhat_test": yhat_test,
            "yproba_test": yproba_test,
        }
        st.session_state["trained_on_file"] = uploaded_file.name
        st.success("✅ Huấn luyện mô hình thành công! Xem kết quả ở tab 'Kết quả huấn luyện & kiểm định mô hình'.")
    except Exception as e:
        st.error(f"Đã xảy ra lỗi khi huấn luyện mô hình: {e}")

# =========================================================
# 6) CÁC TAB NỘI DUNG CHÍNH
# =========================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["🗂️ Tổng quan dữ liệu", "📈 Trực quan hóa dữ liệu", "🎯 Kết quả huấn luyện & kiểm định mô hình", "🔮 Sử dụng mô hình"]
)

# ---------------------------------------------------------
# TAB 1: TỔNG QUAN DỮ LIỆU
# ---------------------------------------------------------
with tab1:
    st.subheader("Kích thước dữ liệu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Số dòng", f"{df.shape[0]:,}")
    c2.metric("Số cột", f"{df.shape[1]:,}")
    c3.metric("Dung lượng file", f"{len(file_bytes) / 1024:.1f} KB")

    st.subheader("Xem dữ liệu thô")
    with st.container(height=300):
        st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Thống kê mô tả (biến đưa vào mô hình)")
    st.dataframe(df[FEATURE_COLS + [TARGET_COL]].describe(), use_container_width=True)

# ---------------------------------------------------------
# TAB 2: TRỰC QUAN HÓA DỮ LIỆU
# ---------------------------------------------------------
with tab2:
    st.subheader("Phân phối biến mục tiêu (PD)")
    target_counts = df[TARGET_COL].value_counts().sort_index().reset_index()
    target_counts.columns = ["PD", "Số lượng"]
    target_counts["PD"] = target_counts["PD"].map({0: "0 - Không rủi ro", 1: "1 - Có rủi ro"})
    fig_target = px.bar(
        target_counts, x="PD", y="Số lượng", color="PD",
        title="Phân phối biến mục tiêu PD",
        color_discrete_sequence=["#2ca02c", "#d62728"],
    )
    fig_target.update_layout(height=380, showlegend=False)
    st.plotly_chart(fig_target, use_container_width=True)

    st.subheader("Phân phối các biến đầu vào")
    default_vars = ["TC1", "NL1", "DK1", "V1"]
    selected_vars = st.multiselect(
        "Chọn biến đầu vào để trực quan hóa (tối đa khuyến nghị 4 biến để xem cân đối)",
        options=FEATURE_COLS,
        default=default_vars,
        help="Có 24 biến đầu vào; chọn tối đa 4 biến để bố cục lưới 2x2 cân đối.",
    )

    vars_to_plot = selected_vars[:4] if selected_vars else default_vars
    cols_row1 = st.columns(2)
    cols_row2 = st.columns(2)
    plot_cols = cols_row1 + cols_row2

    for i, var in enumerate(vars_to_plot[:4]):
        with plot_cols[i]:
            vc = df[var].value_counts().sort_index().reset_index()
            vc.columns = [var, "Số lượng"]
            fig = px.bar(
                vc, x=var, y="Số lượng",
                title=f"Phân phối: {var}",
            )
            fig.update_layout(height=320)
            st.plotly_chart(fig, use_container_width=True)

    if not vars_to_plot:
        st.info("Chọn ít nhất một biến để hiển thị biểu đồ.")

# ---------------------------------------------------------
# TAB 3: KẾT QUẢ HUẤN LUYỆN & KIỂM ĐỊNH MÔ HÌNH
# ---------------------------------------------------------
with tab3:
    if "results" not in st.session_state:
        st.info("👈 Vui lòng bấm nút '🚀 Huấn luyện mô hình' ở thanh bên trái để xem kết quả.")
    else:
        res = st.session_state["results"]
        y_test = res["y_test"]
        yhat_test = res["yhat_test"]
        yproba_test = res["yproba_test"]

        acc = accuracy_score(y_test, yhat_test)
        prec = precision_score(y_test, yhat_test, zero_division=0)
        rec = recall_score(y_test, yhat_test, zero_division=0)
        f1 = f1_score(y_test, yhat_test, zero_division=0)
        try:
            auc = roc_auc_score(y_test, yproba_test)
        except Exception:
            auc = float("nan")

        st.subheader("Chỉ tiêu kiểm định tổng quan")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Accuracy", f"{acc:.3f}")
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("Recall", f"{rec:.3f}")
        m4.metric("F1-score", f"{f1:.3f}")
        m5.metric("ROC-AUC", f"{auc:.3f}" if not np.isnan(auc) else "N/A")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Ma trận nhầm lẫn")
            cm = confusion_matrix(y_test, yhat_test)
            cm_labels = ["0 - Không rủi ro", "1 - Có rủi ro"]
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=cm_labels, y=cm_labels,
                labels=dict(x="Dự báo", y="Thực tế", color="Số lượng"),
                color_continuous_scale="Blues",
            )
            fig_cm.update_layout(height=380)
            st.plotly_chart(fig_cm, use_container_width=True)

        with col_right:
            st.subheader("Đường cong ROC")
            try:
                fpr, tpr, _ = roc_curve(y_test, yproba_test)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc:.3f})"))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ngẫu nhiên", line=dict(dash="dash")))
                fig_roc.update_layout(
                    xaxis_title="Tỷ lệ dương giả (FPR)", yaxis_title="Tỷ lệ dương thật (TPR)", height=380
                )
                st.plotly_chart(fig_roc, use_container_width=True)
            except Exception:
                st.info("Không đủ dữ liệu để vẽ đường cong ROC.")

        st.subheader("Báo cáo phân loại chi tiết (classification report)")
        report_dict = classification_report(y_test, yhat_test, output_dict=True, zero_division=0)
        report_df = pd.DataFrame(report_dict).transpose()
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------
# TAB 4: SỬ DỤNG MÔ HÌNH
# ---------------------------------------------------------
with tab4:
    if "model" not in st.session_state:
        st.info("👈 Vui lòng bấm nút '🚀 Huấn luyện mô hình' ở thanh bên trái trước khi sử dụng mô hình để dự báo.")
    else:
        model = st.session_state["model"]
        feature_cols = st.session_state["feature_cols"]

        mode = st.radio(
            "Chọn chế độ sử dụng",
            options=["Nhập trực tiếp", "Tải file hàng loạt"],
            horizontal=True,
        )

        if mode == "Nhập trực tiếp":
            st.caption("Nhập giá trị cho từng biến đầu vào (thang điểm dựa trên dữ liệu đã tải lên).")
            with st.form("form_predict_single"):
                input_values = {}
                form_cols = st.columns(4)
                for i, col in enumerate(feature_cols):
                    col_min = int(df[col].min())
                    col_max = int(df[col].max())
                    col_median = int(df[col].median())
                    with form_cols[i % 4]:
                        input_values[col] = st.number_input(
                            col,
                            min_value=col_min,
                            max_value=col_max,
                            value=col_median,
                            step=1,
                            help=f"Giá trị trong khoảng [{col_min}, {col_max}] theo dữ liệu hiện có.",
                        )
                submitted = st.form_submit_button("Dự báo", type="primary", use_container_width=True)

            if submitted:
                x_new = pd.DataFrame([input_values])[feature_cols]
                pred = model.predict(x_new)[0]
                proba = model.predict_proba(x_new)[0]
                if pred == 1:
                    st.error(f"⚠️ Kết quả dự báo: **CÓ RỦI RO** (xác suất: {proba[1]:.3f})")
                else:
                    st.success(f"✅ Kết quả dự báo: **KHÔNG RỦI RO** (xác suất: {proba[0]:.3f})")

                p1, p2 = st.columns(2)
                p1.metric("Xác suất không rủi ro (PD=0)", f"{proba[0]:.3f}")
                p2.metric("Xác suất có rủi ro (PD=1)", f"{proba[1]:.3f}")

        else:
            st.caption(f"Tải lên file CSV chứa đúng các cột: {', '.join(feature_cols)}")
            batch_file = st.file_uploader(
                "Tải file dữ liệu mới để dự báo hàng loạt",
                type=["csv"],
                key="batch_predict_uploader",
            )
            if batch_file is not None:
                try:
                    new_df = pd.read_csv(batch_file)
                    missing = validate_columns(new_df, feature_cols)
                    if missing:
                        st.error(f"File thiếu các cột bắt buộc: {', '.join(missing)}")
                    else:
                        x_batch = new_df[feature_cols]
                        preds = model.predict(x_batch)
                        probas = model.predict_proba(x_batch)[:, 1]
                        result_df = new_df.copy()
                        result_df["Dự_báo_PD"] = preds
                        result_df["Xác_suất_rủi_ro"] = probas

                        st.subheader("Kết quả dự báo")
                        with st.container(height=350):
                            st.dataframe(result_df, use_container_width=True)

                        csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "⬇️ Tải kết quả (CSV)",
                            data=csv_bytes,
                            file_name="ket_qua_du_bao.csv",
                            mime="text/csv",
                        )
                except Exception as e:
                    st.error(f"Không thể xử lý file: {e}")
