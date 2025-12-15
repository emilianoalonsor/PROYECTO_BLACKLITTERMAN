import streamlit as st 
import pandas as pd
import numpy as np
from datetime import date

from config import obtener_universo, DIAS_HAB, TINI_RIESGO
from datos import download_precios, calcular_rendimientos, estadis_anuales
from metricas import resumen_metricas
from markowitz import port_min_varianza, port_sharpe, port_rend_objetivo
from black_litterman import black_litterman,view_absoluta,view_relativa,apilar_views

st.set_page_config(
    page_title='Proyecto Seminario de Finanzas',
    page_icon='',
    layout='wide',
)


col1, col2 = st.columns([2,1])
with col1:
    st.title('Seminario de Finanzas--Proyecto final')
    st.markdown('#### Integrantes:')
    st.markdown('-Leslie Rocha Herández')
    st.markdown('-Emiliano Romero Alonso')
with col2:
    st.markdown('#### Información rápida 💡')
    st.write('')

st.sidebar.header('⚙️ Configuración Inicial ')
estrategia = st.sidebar.selectbox('Estrategia de inversión:',['Regiones','Sectores EUA'],)
hoy = date.today()
fecha_inicial = date(hoy.year - 5,hoy.month,hoy.day)
fecha_inicial = st.sidebar.date_input(
    'Fecha de Inicio:',
    value=fecha_inicial,
)
fecha_final = st.sidebar.date_input(
    'Fecha de Fin:',
    value=hoy,
)
st.sidebar.markdown('-----')
tasa_libre_anual = st.sidebar.number_input(
    'Tasa libre anual de riesgo',
    min_value=0.0,
    max_value=0.20,
    value=TINI_RIESGO,
    step=0.005,
)
nivel_var = st.sidebar.slider(
    'Nivel de confianza VaR y CVaR:',
    min_value=0.90,
    max_value=0.99,
    value=0.95,
    step=0.01,
)
st.sidebar.markdown('-----')

etfs, pesos_bench = obtener_universo(estrategia)

st.markdown(f'**Estrategia seleccionada:**  {estrategia}')
st.write('**ETFs del Universo:**',','.join(etfs))


precios = download_precios(etfs, fecha_inicial,fecha_final)
if precios.empty:
    st.error('Error al descargar precios. Vuelve a intentarlo')
    st.stop()
rend = calcular_rendimientos(precios,tipo='log')

if rend.empty:
    st.error('No se pueden calcular los rendimientos.')
    st.stop()

media_anual, cov_anual = estadis_anuales(rend, dias_ano=DIAS_HAB)
num_activos = len(etfs)
num_obs = rend.shape[0]

with col2:
    c1,c2,c3 = st.columns(3)
    c1.metric('Activos',num_activos)
    c2.metric('Observaciones',num_obs)
    c3.metric('Dias Hábiles',DIAS_HAB)
    st.markdown('#### Descarga de datos 📥 ')
    st.write(f'Periodo seleccionado: **{fecha_inicial}** a **{fecha_final}**')
tab_datos, tab_arbitrario, tab_optimi, tab_blitter = st.tabs(
    ['🧾 Datos', '🧮 Portafolio arbitrario', '🎯 Markowitz', '⚠️ Black-Litterman']
)

with tab_datos:
    st.markdown('#### Precios históricos')
    st.line_chart(precios)
    
    col_a,col_b = st.columns(2)
    
    with col_a:
        st.markdown('##### Primeros rendimientos diarios')
        st.dataframe(rend.head())
    
    with col_b:
        st.markdown('##### Medias anuales ETFs')
        st.dataframe(media_anual.to_frame(name='Media anual'))
    
    st.markdown("##### Matriz de covarianzas anual")
    st.dataframe(cov_anual)
  

with tab_arbitrario:
    st.subheader('🎛️ Portafolio personalizado')
    st.write('Ajusta los pesos de cada ETF. La suma se ajusta a 1 automáticamente.')
    cols_sliders = st.columns(3)
    raw_data_user = []
    
    for i, ticker in enumerate(etfs):
        col = cols_sliders[i % 3]
        w = col.slider(
            f'Peso inicial {ticker}',
            min_value = 0.0,
            max_value = 1.0,
            value= float(pesos_bench[i]),
            step=0.01,
        )
        raw_data_user.append(w)
    raw_data_user = np.array(raw_data_user,dtype=float)
    if raw_data_user.sum() == 0:
            pesos_port = np.repeat(1/len(etfs),len(etfs))
    else:
            pesos_port = raw_data_user/raw_data_user.sum()
            
    st.markdown('##### Pesos normalizados del portafolio personalizado')
    df_pesos = pd.DataFrame({'Ticker': etfs, 'Peso': pesos_port,}).set_index('Ticker')
    st.dataframe(df_pesos)
    
    metricas_arbi, rend_port_arbi, rend_arbi_bench = resumen_metricas(
        rendimientos=rend,
        pesos_port = pesos_port,
        pesos_bench=pesos_bench,
        tasa_libre_anual=tasa_libre_anual,
        nivel_var=nivel_var,
        dias_ano=DIAS_HAB,
    )
    st.markdown('##### Métricas del portafolio arbitrario')
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    col_m1.metric(
        'Rendimiento anual esperado',
        f"{metricas_arbi['Rendimiento anual esperado']:.2%}"
    )
    col_m2.metric(
        'Volatilidad anual',
        f"{metricas_arbi['Volatilidad anual']:.2%}",
        
    )
    col_m3.metric(
        'Sharpe',
        f"{metricas_arbi['Sharpe']:.2f}",
        
    )
    col_m4.metric(
        'Max Drawdown',
        f"{metricas_arbi['Max Drawdown']:.2%}",
        
    )
    st.dataframe(pd.DataFrame(metricas_arbi, index=['Valor']).T)
    st.markdown('##### Rendimiento acumulado portafolio vs benchmark')
    acum_port = (1 + rend_port_arbi).cumprod()
    acum_bench = (1+ rend_arbi_bench).cumprod()
    df_acum = pd.concat(
        [acum_port.rename('Portafolio'), acum_bench.rename('Benchmark')],
        axis=1,
    )
    st.line_chart(df_acum)

with tab_optimi:
    st.subheader('🧬 Portafolio Markowitz')
    
    tipo_opt = st.selectbox(
        'Tipo de optimización:',
        ['Minima Varianza','Máximo Sharpe','Rendimiento objetivo'],
    )
    rend_obj = None
    if tipo_opt == 'Rendimiento objetivo':
        rend_obj = st.number_input(
            'Rendimiento objetivo anual',
            min_value=-0.50,
            max_value=0.50,
            value=float(media_anual.mean()),
            step=0.01,
        )
    if st.button('Calcular portafolio óptimo'):
        if tipo_opt == 'Minima Varianza':
            w_opt = port_min_varianza(media_anual,cov_anual)
        elif tipo_opt == 'Máximo Sharpe':
            w_opt = port_sharpe(media_anual,cov_anual,tasa_libre_anual)
        else:
            w_opt = port_rend_objetivo(
                media_anual,cov_anual,rend_obj
            )
        w_opt = np.array(w_opt, dtype=float)
        w_opt = w_opt/w_opt.sum()
        
        st.markdown('##### Pesos del portafolio óptimo')
        df_wopt = pd.DataFrame({'Ticker':etfs, 'Peso óptimo':w_opt}).set_index('Ticker')
        st.dataframe(df_wopt)    
        
        metricas_opt, rend_port_opt, rend_bench_opt = resumen_metricas(
            rendimientos=rend,
            pesos_port = w_opt,
            pesos_bench=pesos_bench,
            tasa_libre_anual=tasa_libre_anual,
            nivel_var=nivel_var,
            dias_ano=DIAS_HAB,
        )
        st.markdown('##### Métricas del portafolio óptimo')
                
        c1, c2, c3, c4 = st.columns(4)
        c1.metric(
            "Rend. anual esperado",
            f"{metricas_opt['Rendimiento anual esperado']:.2%}",
        )
        c2.metric(
            "Volatilidad anual",
            f"{metricas_opt['Volatilidad anual']:.2%}",
        )
        c3.metric(
            "Sharpe",
            f"{metricas_opt['Sharpe']:.2f}",
        )
        c4.metric(
            "Max Drawdown",
            f"{metricas_opt['Max Drawdown']:.2%}",
        )

        st.dataframe(pd.DataFrame(metricas_opt, index=["Valor"]).T)
        
        st.markdown("##### Rendimiento acumulado: óptimo vs benchmark")
        acum_port_o = (1 + rend_port_opt).cumprod()
        acum_bench_o = (1 + rend_bench_opt).cumprod()
        df_acum_o = pd.concat(
            [acum_port_o.rename("Portafolio óptimo"), acum_bench_o.rename("Benchmark")],
            axis=1,
        )
        st.line_chart(df_acum_o)
        
        st.markdown('##### Distribución de pesos del portafolio óptimo')
        st.bar_chart(df_wopt)

with tab_blitter:
    st.subheader("⚠️ Black–Litterman")

   
    if "bl_rowsP" not in st.session_state:
        st.session_state.bl_rowsP = []
    if "bl_valsQ" not in st.session_state:
        st.session_state.bl_valsQ = []
    if "bl_conf" not in st.session_state:
        st.session_state.bl_conf = []
    if "bl_result" not in st.session_state:
        st.session_state.bl_result = None  

    st.markdown("### 1) Definir views")

    cL, cR = st.columns([1, 1])

    with cL:
        tipo_view = st.selectbox(
            "Tipo de view",
            ["Relativa (Activo favorecido – Activo desfavorecido)", "Absoluta (Activo favorecido)"],
            key="bl_tipo_view",
        )

        if tipo_view.startswith("Relativa"):
            long_t = st.selectbox("Long (ganador)", etfs, key="bl_long")
            short_t = st.selectbox("Short (perdedor)", etfs, key="bl_short")
            q = st.number_input(
                "Ventaja esperada del Long sobre el Short anual",
                min_value=-1.0, max_value=1.0, value=0.02, step=0.005,
                key="bl_q_rel",
            )
        else:
            t = st.selectbox("Activo", etfs, key="bl_abs_ticker")
            q = st.number_input(
                "Q (retorno anual en exceso)",
                min_value=-1.0, max_value=1.0, value=0.02, step=0.005,
                key="bl_q_abs",
            )

        conf_k = st.slider(
            "Confianza de esta view",
            min_value=0.10, max_value=1.00, value=0.60, step=0.05,
            key="bl_conf_k",
        )

        btn_add, btn_clear = st.columns(2)
        with btn_add:
            if st.button("➕ Agregar view", key="bl_add"):
                if tipo_view.startswith("Relativa"):
                    row, q_val = view_relativa(etfs, long_t, short_t, q)
                    preview = f"Relativa: {long_t} – {short_t} = {q_val:.2%} | conf={conf_k:.2f}"
                else:
                    row, q_val = view_absoluta(etfs, t, q)
                    preview = f"Absoluta: {t} = {q_val:.2%} | conf={conf_k:.2f}"

                st.session_state.bl_rowsP.append(row)
                st.session_state.bl_valsQ.append(float(q_val))
                st.session_state.bl_conf.append(float(conf_k))

                st.session_state.bl_result = None
                st.success("View agregada ✅")
                st.caption(f"Vista previa: {preview}")

        with btn_clear:
            if st.button("🧹 Limpiar views", key="bl_clear"):
                st.session_state.bl_rowsP = []
                st.session_state.bl_valsQ = []
                st.session_state.bl_conf = []
                st.session_state.bl_result = None
                st.warning("Views limpiadas.")

    with cR:
        st.markdown("### 2) Views actuales")
        if len(st.session_state.bl_rowsP) == 0:
            st.info("Aún no has agregado views.")
        else:
            P, Q = apilar_views(st.session_state.bl_rowsP, st.session_state.bl_valsQ)
            dfP = pd.DataFrame(P, columns=etfs)
            dfQ = pd.DataFrame({"Q (anual exceso)": Q, "conf": st.session_state.bl_conf})
            st.dataframe(dfP, use_container_width=True)
            st.dataframe(dfQ, use_container_width=True)

    st.markdown("### 3) Ejecutar Black–Litterman")

    cparams, cbtn = st.columns([2, 1])

    with cparams:
        tau = st.number_input("τ (tau)", min_value=0.001, max_value=0.200, value=0.025, step=0.001, key="bl_tau")
        delta = st.number_input("δ (delta)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="bl_delta")

    with cbtn:
        run_bl = st.button("Calcular BL", key="bl_run")

    if run_bl:
        if len(st.session_state.bl_rowsP) == 0:
            P = None
            Q = None
            conf_vec = None
        else:
            P, Q = apilar_views(st.session_state.bl_rowsP, st.session_state.bl_valsQ)
            conf_vec = np.array(st.session_state.bl_conf, dtype=float)

        mu_excess, mu_total, Sigma_post, pi, Omega, delta_out, tau_out = black_litterman(
            Sigma=cov_anual,
            w_mkt=pesos_bench,
            rf_anual=tasa_libre_anual,
            mu_anual=media_anual,   
            P=P,
            Q=Q,
            conf=conf_vec,
            tau=float(tau),
            delta=float(delta),
        )

      
        st.session_state.bl_result = {
            "mu_bl": np.array(mu_total, dtype=float),  
            "pi": np.array(pi, dtype=float),
            "Omega": np.array(Omega, dtype=float),
            "tau": float(tau_out),
            "delta": float(delta_out),
            "P": None if P is None else np.array(P, dtype=float),
            "Q": None if Q is None else np.array(Q, dtype=float),
        }

        st.success("Black–Litterman calculado ✅")

  
    if st.session_state.bl_result is not None:
        res = st.session_state.bl_result
        mu_bl = res["mu_bl"]

        st.markdown("### Parámetros usados")
        p1, p2, p3 = st.columns(3)
        p1.metric("τ (tau)", f"{res['tau']:.3f}")
        p2.metric("δ (delta)", f"{res['delta']:.3f}")
        p3.metric("K (views)", 0 if res["P"] is None else int(res["P"].shape[0]))

        st.markdown("### μ_BL (retornos anuales esperados)")
        df_mu = pd.DataFrame({"μ_BL": mu_bl}, index=etfs)
        st.dataframe(df_mu.style.format("{:.2%}"), use_container_width=True)

    
        st.markdown("## 5) Markowitz usando μ_BL)")

        with st.form("form_opt_bl"):
            tipo_opt_bl = st.selectbox(
                "Optimización (BL):",
                ["Minima Varianza", "Máximo Sharpe", "Rendimiento objetivo"],
                key="bl_opt_tipo",
            )

            rend_obj_bl = None
            if tipo_opt_bl == "Rendimiento objetivo":
                rend_obj_bl = st.number_input(
                    "Rendimiento objetivo anual (μ_BL)",
                    min_value=-0.50, max_value=0.50,
                    value=float(np.mean(mu_bl)),
                    step=0.01,
                    key="bl_rend_obj",
                )

            submit_opt = st.form_submit_button("Calcular portafolio óptimo con μ_BL")

        if submit_opt:
            if tipo_opt_bl == "Minima Varianza":
                w_bl = port_min_varianza(mu_bl, cov_anual)
            elif tipo_opt_bl == "Máximo Sharpe":
                w_bl = port_sharpe(mu_bl, cov_anual, tasa_libre_anual)
            else:
                w_bl = port_rend_objetivo(mu_bl, cov_anual, rend_obj_bl)

            w_bl = np.array(w_bl, dtype=float)
            w_bl = w_bl / w_bl.sum()

            st.markdown("### Pesos óptimos")
            df_wbl = pd.DataFrame({"Peso": w_bl}, index=etfs)
            st.dataframe(df_wbl.style.format("{:.2%}"), use_container_width=True)
            
            st.markdown("##### Pesos finales Black–Litterman")

            df_plot_bl = df_wbl.copy()
            df_plot_bl = df_plot_bl.sort_values("Peso", ascending=False)

            st.bar_chart(df_plot_bl)
            
            metricas_bl, rend_port_bl, rend_bench_bl = resumen_metricas(
                rendimientos=rend,
                pesos_port=w_bl,
                pesos_bench=pesos_bench,
                tasa_libre_anual=tasa_libre_anual,
                nivel_var=nivel_var,
                dias_ano=DIAS_HAB,
            )

            st.markdown("### Métricas del portafolio")
            st.dataframe(pd.DataFrame(metricas_bl, index=["Valor"]).T, use_container_width=True)
