import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import plotly.io as pio
import io

# 页面配置
st.set_page_config(page_title="弱电解质电离平衡常数处理", layout="wide")
st.title("🧪 电导法测定弱电解质的电离平衡常数")
st.markdown("根据醋酸溶液浓度和电导率，计算摩尔电导率、电离度、平衡常数，并绘制相关图表。")

# 固定参数
LAMBDA_INF_DEFAULT = 390.72e-4  # S·m²·mol⁻¹, 25°C

# 侧边栏参数设置
st.sidebar.header("⚙️ 参数设置")
use_correction = st.sidebar.checkbox("启用电极常数校正", value=False)
lambda_inf = st.sidebar.number_input(
    "无限稀释摩尔电导率 Λₘ∞ (×10⁻⁴ S·m²·mol⁻¹)",
    value=float(LAMBDA_INF_DEFAULT * 1e4),
    format="%.2f"
) * 1e-4

st.sidebar.markdown("---")
if use_correction:
    st.sidebar.subheader("电极常数校正")
    kcl_theory = st.sidebar.number_input("KCl 理论电导率 (mS/cm, 25°C 0.01M)", value=1.4083, format="%.4f")
    kcl_measured = st.sidebar.number_input("KCl 实测电导率 (mS/cm)", value=1.4283, format="%.4f")
    if kcl_measured != 0:
        correction_factor = kcl_theory / kcl_measured
    else:
        correction_factor = 1.0
    st.sidebar.write(f"校正因子 = {correction_factor:.4f}")
else:
    correction_factor = 1.0

# 数据输入方式
st.subheader("📥 数据输入")
data_source = st.radio("选择数据输入方式", ["表格编辑", "上传 CSV 文件"])

if data_source == "表格编辑":
    default_data = {
        "浓度 (mol/L)": [0.1, 0.05, 0.025, 0.0125, 0.00625, 0.003125],
        "电导率 (mS/cm)": [0.53, 0.25, 0.18, 0.14, 0.09, 0.06]
    }
    df_input = pd.DataFrame(default_data)
    edited_df = st.data_editor(df_input, num_rows="dynamic", use_container_width=True)
else:
    uploaded_file = st.file_uploader("上传 CSV 文件 (两列: 浓度 (mol/L), 电导率 (mS/cm))", type="csv")
    if uploaded_file is not None:
        edited_df = pd.read_csv(uploaded_file)
        st.dataframe(edited_df)
    else:
        st.info("请上传 CSV 文件")
        st.stop()

if edited_df.empty:
    st.warning("请至少输入一组数据")
    st.stop()

# 计算按钮
if st.button("🚀 开始计算", type="primary"):
    # 应用校正因子
    conductivity_raw = edited_df.iloc[:, 1].values
    conductivity_corrected = conductivity_raw * correction_factor
    edited_df["校正后电导率 (mS/cm)"] = conductivity_corrected
    edited_df["电导率 (S/m)"] = conductivity_corrected * 0.1

    # 计算
    conc_molL = edited_df.iloc[:, 0].values
    conc_molm3 = conc_molL * 1000
    kappa_Sm = edited_df["电导率 (S/m)"].values

    with np.errstate(divide='ignore', invalid='ignore'):
        Lambda_m = np.where(conc_molm3 > 0, kappa_Sm / conc_molm3, np.nan)
        alpha = Lambda_m / lambda_inf
        Kc = (conc_molL * alpha**2) / (1 - alpha)

    # 结果表格
    result_df = pd.DataFrame({
        "浓度 (mol/L)": conc_molL,
        "电导率 (mS/cm)": conductivity_corrected,
        "摩尔电导率 Λₘ (×10⁻⁴ S·m²·mol⁻¹)": Lambda_m * 1e4,
        "电离度 α": alpha,
        "平衡常数 Kc (×10⁻⁵)": Kc * 1e5
    })

    st.subheader("📊 计算结果")
    st.dataframe(result_df, use_container_width=True)

    # 统计 Kc
    Kc_valid = Kc[np.isfinite(Kc)]
    if len(Kc_valid) > 0:
        Kc_mean = np.mean(Kc_valid)
        Kc_std = np.std(Kc_valid, ddof=1) if len(Kc_valid) > 1 else 0.0
        st.success(f"**平均 Kc = {Kc_mean:.3e} mol/L**")
        if len(Kc_valid) > 1:
            st.info(f"标准差 = {Kc_std:.3e} mol/L, 相对标准差 = {Kc_std/Kc_mean*100:.2f}%")
        theoretical = 1.75e-5
        st.write(f"参考理论值 (25°C) : 1.75×10⁻⁵, 相对误差 = {abs(Kc_mean-theoretical)/theoretical*100:.2f}%")
    else:
        st.warning("无法计算有效的Kc值")
        Kc_mean = np.nan

    # 绘图（存储到变量以便报告使用）
    st.subheader("📈 图表分析")
    tab1, tab2, tab3, tab4 = st.tabs(["Λₘ - c", "α - c", "Kc 分布", "Ostwald 稀释定律验证"])

    fig1 = fig2 = fig3 = fig4 = None
    with tab1:
        if len(conc_molL) >= 2:
            fig1 = px.scatter(result_df, x="浓度 (mol/L)", y="摩尔电导率 Λₘ (×10⁻⁴ S·m²·mol⁻¹)",
                              title="摩尔电导率随浓度的变化", log_x=True, trendline=None)
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.info("至少需要2个数据点才能绘制此图")

    with tab2:
        if len(conc_molL) >= 2:
            fig2 = px.scatter(result_df, x="浓度 (mol/L)", y="电离度 α",
                              title="电离度随浓度的变化", log_x=True, trendline=None)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("至少需要2个数据点才能绘制此图")

    with tab3:
        if len(conc_molL) >= 1:
            fig3 = px.bar(result_df, x="浓度 (mol/L)", y="平衡常数 Kc (×10⁻⁵)",
                          title="各浓度下的 Kc 值", text_auto=True)
            if not np.isnan(Kc_mean):
                fig3.add_hline(y=Kc_mean*1e5, line_dash="dash", line_color="red",
                               annotation_text=f"平均值 = {Kc_mean:.2e}")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("数据点不足，无法绘制Kc分布图")

    with tab4:
        x_ost = 1.0 / Lambda_m
        y_ost = conc_molL * Lambda_m
        valid_idx = np.isfinite(x_ost) & np.isfinite(y_ost)
        if np.sum(valid_idx) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_ost[valid_idx], y_ost[valid_idx])
            fig4 = px.scatter(x=x_ost[valid_idx], y=y_ost[valid_idx],
                              labels={"x": "1/Λₘ (mol·m⁻³·S⁻¹·m²)", "y": "c·Λₘ (S·m²·mol⁻¹·mol/L)"},
                              title="Ostwald 稀释定律验证")
            x_line = np.array([min(x_ost[valid_idx]), max(x_ost[valid_idx])])
            y_line = slope * x_line + intercept
            fig4.add_trace(go.Scatter(x=x_line, y=y_line, mode="lines", name=f"拟合线 (R²={r_value**2:.3f})"))
            st.plotly_chart(fig4, use_container_width=True)
            st.write(f"拟合方程: cΛₘ = {slope:.2e} × (1/Λₘ) + {intercept:.2e}")
        else:
            st.info("至少需要2个有效数据点才能进行Ostwald稀释定律验证")

    # 导出和打印功能
    st.subheader("💾 导出与打印")
    col1, col2 = st.columns(2)
    with col1:
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 下载计算结果 CSV", csv_data, "weak_acid_results.csv", "text/csv")
    with col2:
        # 生成 HTML 报告文件并下载
        if st.button("📄 导出报告为 HTML (可打印为 PDF)"):
            # 构建报告HTML
            html_parts = []
            html_parts.append(f"""
            <html>
            <head>
                <title>电导法测定弱电解质电离平衡常数报告</title>
                <meta charset="UTF-8">
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #34495e; border-bottom: 1px solid #ddd; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                    th {{ background-color: #f2f2f2; }}
                    .chart {{ margin: 30px 0; }}
                </style>
            </head>
            <body>
                <h1>电导法测定弱电解质的电离平衡常数实验报告</h1>
                <p><strong>生成时间：</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <h2>1. 原始数据与计算结果</h2>
                {result_df.to_html(index=False)}
                <h2>2. 统计结果</h2>
                <p>平均 Kc = {Kc_mean:.3e} mol/L</p>
                <p>参考理论值 (25°C) : 1.75×10⁻⁵ mol/L, 相对误差 = {abs(Kc_mean-theoretical)/theoretical*100:.2f}%</p>
                <h2>3. 图表</h2>
            """)
            # 嵌入图表
            if fig1 is not None:
                html_parts.append('<div class="chart"><h3>摩尔电导率随浓度的变化</h3>')
                html_parts.append(pio.to_html(fig1, full_html=False, include_plotlyjs='cdn'))
                html_parts.append('</div>')
            if fig2 is not None:
                html_parts.append('<div class="chart"><h3>电离度随浓度的变化</h3>')
                html_parts.append(pio.to_html(fig2, full_html=False, include_plotlyjs='cdn'))
                html_parts.append('</div>')
            if fig3 is not None:
                html_parts.append('<div class="chart"><h3>各浓度下的 Kc 值</h3>')
                html_parts.append(pio.to_html(fig3, full_html=False, include_plotlyjs='cdn'))
                html_parts.append('</div>')
            if fig4 is not None:
                html_parts.append('<div class="chart"><h3>Ostwald 稀释定律验证</h3>')
                html_parts.append(pio.to_html(fig4, full_html=False, include_plotlyjs='cdn'))
                html_parts.append('</div>')
            html_parts.append("</body></html>")
            full_html = "".join(html_parts)
            # 提供下载
            st.download_button(
                label="📄 点击下载 HTML 报告",
                data=full_html.encode("utf-8"),
                file_name="weak_acid_report.html",
                mime="text/html"
            )
            st.info("下载后，用浏览器打开该 HTML 文件，然后按 Ctrl+P (或 Cmd+P) 选择「另存为 PDF」即可生成 PDF 报告。")

    st.markdown("---")
    st.caption("实验原理：Λₘ = κ / c, α = Λₘ / Λₘ∞, Kc = c·α²/(1-α)")
else:
    st.info("👆 请输入或上传数据后，点击「开始计算」按钮查看结果")
