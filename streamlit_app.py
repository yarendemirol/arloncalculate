import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

st.set_page_config(layout="wide", page_title="ARLON & ARAS Methods Application")

# --- ARLON METODU FONKSİYONU ---
def run_arlon_method_with_intermediate_results(initial_decision_matrix, criterion_weights, benefit_criteria, cost_criteria, zeta_value):
    ## Adım 2-2: İki Aşamalı Logaritmik Normalizasyon ve Heron Ortalaması Agregasyonu

    # Ensure all values in the decision matrix are positive for logarithm operations
    if (initial_decision_matrix <= 0).any().any():
        st.error("Decision matrix contains zero or negative values. Logarithm cannot be applied. Please ensure all performance values are positive.")
        st.stop()

    normalized_log1_df = pd.DataFrame(index=initial_decision_matrix.index, columns=initial_decision_matrix.columns)
    for col in initial_decision_matrix.columns:
        prod_xij = initial_decision_matrix[col].prod()
        # Handle cases where prod_xij might be 1 (if all xij for a criterion are 1) to avoid log(1)=0 division error
        if prod_xij == 1 and (col in benefit_criteria or (col in cost_criteria and len(initial_decision_matrix) - 1 == 0)):
            st.warning(f"Kriter '{col}' için tüm alternatif değerleri 1 olduğu için logaritma tabanı sıfır olabilir. Bu kriter için normalizasyon 1 olarak ayarlandı.")
            normalized_log1_df[col] = 1.0 # Or appropriate handling if all values are 1
            continue # Skip to next column

        if col in benefit_criteria:
            normalized_log1_df[col] = np.log(initial_decision_matrix[col]) / np.log(prod_xij)
        elif col in cost_criteria:
            m = len(initial_decision_matrix)
            # Avoid division by zero if m-1 is zero (i.e., only one alternative)
            if m - 1 == 0:
                normalized_log1_df[col] = 1 - (np.log(initial_decision_matrix[col])) # Only one alternative, so prod_xij is xij itself. This simplifies the formula.
            else:
                normalized_log1_df[col] = 1 - (np.log(initial_decision_matrix[col]) / (np.log(prod_xij) * (m - 1)))


    normalized_log2_df = pd.DataFrame(index=initial_decision_matrix.index, columns=initial_decision_matrix.columns)
    for col in initial_decision_matrix.columns:
        sum_log2_xij = np.log2(initial_decision_matrix[col]).sum()
        # Handle cases where sum_log2_xij might be 0 (e.g., if all xij for a criterion are 1)
        if sum_log2_xij == 0:
            st.warning(f"Kriter '{col}' için log2 toplamı sıfır olduğu için bu kriterin normalizasyonu 1 olarak ayarlandı.")
            normalized_log2_df[col] = 1.0 # Or appropriate handling
            continue # Skip to next column

        if col in benefit_criteria:
            normalized_log2_df[col] = np.log2(initial_decision_matrix[col]) / sum_log2_xij
        elif col in cost_criteria:
            normalized_log2_df[col] = 1 - (np.log2(initial_decision_matrix[col]) / sum_log2_xij)

    aggregated_normalized_df = pd.DataFrame(index=initial_decision_matrix.index, columns=initial_decision_matrix.columns)
    for col in initial_decision_matrix.columns:
        A = normalized_log1_df[col]
        B = normalized_log2_df[col]
        # Use np.maximum(0, ...) to handle potential floating point inaccuracies leading to slightly negative values
        aggregated_normalized_df[col] = ((1 - zeta_value) * np.sqrt(np.maximum(0, A * B))) + (zeta_value * ((A + B) / 2))

    ## Adım 2-3: Ağırlıklı Agregasyon (Weighted Aggegated Normalization)
    weighted_aggregated_normalized_df = aggregated_normalized_df.copy()
    for col in aggregated_normalized_df.columns:
        if col in criterion_weights.index:
            weighted_aggregated_normalized_df[col] = aggregated_normalized_df[col] * criterion_weights[col]

    ## Adım 2-4: Maliyet ve Fayda Kriterlerinin Ayrı Ayrı Toplanması
    # Ensure cost_criteria and benefit_criteria are subsets of initial_decision_matrix.columns
    current_cost_criteria = [c for c in cost_criteria if c in weighted_aggregated_normalized_df.columns]
    current_benefit_criteria = [c for c in benefit_criteria if c in weighted_aggregated_normalized_df.columns]

    cost_sums = weighted_aggregated_normalized_df[current_cost_criteria].sum(axis=1) if current_cost_criteria else pd.Series(0.0, index=initial_decision_matrix.index)
    benefit_sums = weighted_aggregated_normalized_df[current_benefit_criteria].sum(axis=1) if current_benefit_criteria else pd.Series(0.0, index=initial_decision_matrix.index)

    result_df = pd.DataFrame({
        'Cost_Sum (ℂ_i)': cost_sums,
        'Benefit_Sum (B_i)': benefit_sums
    }, index=initial_decision_matrix.index)

    ## Adım 2-5: Alternatiflerin Nihai Sıralaması
    total_criteria_count = len(current_benefit_criteria) + len(current_cost_criteria)
    if total_criteria_count == 0:
        psi_value = 0.5 # Default or handle as error if no criteria selected
    else:
        psi_value = len(current_benefit_criteria) / total_criteria_count

    # Handle cases where benefit_sums or cost_sums might be zero leading to log(0)
    final_ranking_scores = []
    for idx in result_df.index:
        b_i = result_df.loc[idx, 'Benefit_Sum (B_i)']
        c_i = result_df.loc[idx, 'Cost_Sum (ℂ_i)']

        # Ensure that base for power calculation is non-negative
        term_benefit = b_i ** psi_value if b_i >= 0 else 0
        term_cost = c_i ** (1 - psi_value) if c_i >= 0 else 0

        final_ranking_scores.append(term_benefit + term_cost)

    result_df['Final_Ranking_Score (ℝ_i)'] = final_ranking_scores

    final_ranking = result_df.sort_values(by='Final_Ranking_Score (ℝ_i)', ascending=False)
    final_ranking['Ranking'] = np.arange(1, len(final_ranking) + 1)

    return final_ranking[['Final_Ranking_Score (ℝ_i)', 'Ranking']], normalized_log1_df, normalized_log2_df, aggregated_normalized_df, weighted_aggregated_normalized_df, result_df[['Cost_Sum (ℂ_i)', 'Benefit_Sum (B_i)']]


# --- ARAS METODU FONKSİYONU ---
def run_aras_method(initial_decision_matrix, criterion_weights, benefit_criteria, cost_criteria):
    # Step 1: Normalize the decision matrix
    normalized_matrix = initial_decision_matrix.copy().astype(float)
    for col in initial_decision_matrix.columns:
        if col in benefit_criteria:
            normalized_matrix[col] = initial_decision_matrix[col] / initial_decision_matrix[col].sum()
        elif col in cost_criteria:
            normalized_matrix[col] = (1 / initial_decision_matrix[col]) / (1 / initial_decision_matrix[col]).sum()

    # Step 2: Calculate the weighted normalized matrix
    weighted_normalized_matrix = normalized_matrix.copy()
    for col in initial_decision_matrix.columns:
        weighted_normalized_matrix[col] = normalized_matrix[col] * criterion_weights[col]

    # Step 3: Calculate the optimality function S_i for each alternative
    optimality_function_Si = weighted_normalized_matrix.sum(axis=1)

    # Step 4: Calculate the degree of optimality K_i
    # Find the maximum S_i (ideal alternative)
    S_max = optimality_function_Si.max()

    # Avoid division by zero if S_max is 0 (unlikely with positive values, but good practice)
    if S_max == 0:
        st.warning("Maximum optimality function value (S_max) is zero. ARAS ranking might be affected.")
        degree_of_optimality_Ki = pd.Series(0.0, index=initial_decision_matrix.index)
    else:
        degree_of_optimality_Ki = optimality_function_Si / S_max

    # Step 5: Rank the alternatives
    aras_ranking_df = pd.DataFrame({
        'ARAS_Score (K_i)': degree_of_optimality_Ki
    }, index=initial_decision_matrix.index)

    final_ranking = aras_ranking_df.sort_values(by='ARAS_Score (K_i)', ascending=False)
    final_ranking['Ranking'] = np.arange(1, len(final_ranking) + 1)

    return final_ranking[['ARAS_Score (K_i)', 'Ranking']]


# --- STREAMLIT UYGULAMASI BAŞLANGICI ---
st.title("ARLON ve ARAS Çok Kriterli Karar Verme Yöntemleri")
st.markdown("Bu uygulama, ARLON (Average Relative Logarithm Normalization) ve ARAS (Additive Ratio ASsessment) yöntemlerini kullanarak alternatifleri değerlendirmenizi ve sıralamanızı sağlar.")

# --- DİNAMİK VERİ GİRİŞİ ---
st.header("1. Veri Girişi")

# Alternatifler
st.subheader("Alternatifler")
num_alternatives = st.number_input("Değerlendirmek istediğiniz alternatif sayısı:", min_value=1, value=3, step=1)
alternatives = []
for i in range(num_alternatives):
    alt_name = st.text_input(f"Alternatif {i+1} Adı:", value=f"Alternatif {i+1}")
    alternatives.append(alt_name)

# Kriterler
st.subheader("Kriterler")
num_criteria = st.number_input("Değerlendirme kriteri sayısı:", min_value=1, value=9, step=1) # Default 9 kriter
criteria = []
criterion_types = {}
for i in range(num_criteria):
    col1, col2 = st.columns(2)
    with col1:
        # Örnek kriter isimleri ve tipleriyle önceden dolduralım
        default_crit_name = f"Kriter {i+1}"
        default_crit_type = 'Fayda' # Varsayılan Fayda

        # Sizin verdiğiniz orijinal kriterler için varsayılan değerleri ayarlayalım
        if i == 0:
            default_crit_name = 'Lojistik Maliyeti'
            default_crit_type = 'Maliyet'
        elif i == 1: default_crit_name = 'Teslimat Süresi'
        elif i == 2: default_crit_name = 'Müşteri Memnuniyeti'
        elif i == 3: default_crit_name = 'İade Süreci Verimliliği'
        elif i == 4: default_crit_name = 'Sipariş Doğruluğu'
        elif i == 5: default_crit_name = 'Takip Sistemi Etkinliği'
        elif i == 6: default_crit_name = 'Envanter Yönetimi Verimliliği'
        elif i == 7: default_crit_name = 'Son Mil Teslimat Çözümleri'
        elif i == 8: default_crit_name = 'Lojistikte Sürdürülebilirlik'


        crit_name = st.text_input(f"Kriter {i+1} Adı:", value=default_crit_name, key=f"crit_name_{i}")
        if crit_name:
            criteria.append(crit_name)
    with col2:
        crit_type = st.radio(f"Kriter '{crit_name}' Tipi:", ('Fayda', 'Maliyet'), index=0 if default_crit_type == 'Fayda' else 1, key=f"crit_type_{i}")
        criterion_types[crit_name] = crit_type

benefit_criteria = [c for c, t in criterion_types.items() if t == 'Fayda']
cost_criteria = [c for c, t in criterion_types.items() if t == 'Maliyet']

# Kriter Ağırlıkları
st.subheader("Kriter Ağırlıkları (toplamı 1'e yakın olmak üzere otomatik normalize edilecektir)")
st.info("Kriter ağırlıklarını daha fazla ondalık hane hassasiyetiyle girebilirsiniz.")
criterion_weights_input = {}
if criteria:
    weight_cols = st.columns(len(criteria))
    # Sizin orijinal anket puanlarınızdan ağırlık ortalamalarını hesaplayıp önceden dolduralım
    original_criterion_raw_scores = {
        'Lojistik Maliyeti': [10, 10, 10, 9, 9, 9, 8, 7, 6, 6],
        'Teslimat Süresi': [10, 10, 10, 9, 9, 9, 8, 7, 7, 7],
        'Müşteri Memnuniyeti': [10, 10, 10, 10, 9, 8, 8, 7, 7, 6],
        'İade Süreci Verimliliği': [10, 10, 9, 9, 8, 8, 7, 7, 7, 5],
        'Sipariş Doğruluğu': [10, 10, 10, 10, 9, 9, 9, 8, 8, 7],
        'Takip Sistemi Etkinliği': [10, 10, 10, 10, 9, 8, 8, 8, 6, 5],
        'Envanter Yönetimi Verimliliği': [10, 10, 10, 9, 9, 9, 8, 8, 8, 6],
        'Son Mil Teslimat Çözümleri': [10, 9, 9, 9, 8, 7, 7, 6, 5, 4],
        'Lojistikte Sürdürülebilirlik': [10, 10, 9, 8, 7, 7, 5, 4, 4, 3]
    }
    calculated_avg_weights = {crit: np.mean(scores) for crit, scores in original_criterion_raw_scores.items()}


    for i, crit in enumerate(criteria):
        with weight_cols[i]:
            # Eğer kriter orijinal listede varsa, ortalama puanını varsayılan değer olarak kullan
            default_weight_value = calculated_avg_weights.get(crit, 1.0 / len(criteria)) # Bulamazsa eşit dağıt
            weight = st.number_input(
                f"{crit} Ağırlığı:",
                min_value=0.0,
                max_value=100.0, # Max değeri, ortalama puanlar 10'a kadar çıkabildiği için ayarladık
                value=float(default_weight_value),
                step=0.000001, # Daha küçük adım boyutu
                format="%.6f", # 6 ondalık hane hassasiyeti
                key=f"weight_{crit}"
            )
            criterion_weights_input[crit] = weight
else:
    st.warning("Lütfen önce kriterleri giriniz.")

# Kriter ağırlıklarını normalize et
if criterion_weights_input:
    sum_weights = sum(criterion_weights_input.values())
    if sum_weights > 0:
        criterion_weights_normalized = {k: v / sum_weights for k, v in criterion_weights_input.items()}
        criterion_weights_series = pd.Series(criterion_weights_normalized)
        st.info(f"Girilen ağırlıkların toplamı: {sum_weights:.6f}. Ağırlıklar otomatik olarak normalize edildi.")
        st.dataframe(pd.DataFrame({'Kriter': criterion_weights_series.index, 'Normalize Ağırlık': criterion_weights_series.values}).set_index('Kriter').T, use_container_width=True)

    else:
        st.warning("Kriter ağırlıklarının toplamı sıfır olamaz. Lütfen pozitif değerler girin.")
        criterion_weights_series = pd.Series()
else:
    criterion_weights_series = pd.Series()


# Karar Matrisi
st.subheader("Karar Matrisi (Alternatiflerin Kriterlere Göre Performans Değerleri)")
st.markdown("Her bir alternatif için her bir kriterdeki performans değerini giriniz. Hassasiyet için daha fazla ondalık hane kullanabilirsiniz.")

if alternatives and criteria:
    # Sizin orijinal şirket verilerinizden ortalamaları hesaplayıp varsayılan değer olarak kullanalım
    original_company_raw_scores = {
        'Trendyol': {
            'Teslimat Süresi': [4, 7, 8, 8, 9, 9, 9, 10, 10, 10],
            'Müşteri Memnuniyeti': [10, 10, 10, 10, 9, 9, 8, 7, 7, 4],
            'İade Süreci Verimliliği': [10, 10, 10, 9, 9, 8, 8, 8, 7, 4],
            'Sipariş Doğruluğu': [10, 10, 9, 9, 9, 8, 8, 8, 5, 4],
            'Takip Sistemi Etkinliği': [10, 9, 9, 9, 9, 8, 8, 8, 8, 3],
            'Envanter Yönetimi Verimliliği': [9, 8, 8, 8, 8, 8, 8, 6, 6, 4],
            'Son Mil Teslimat Çözümleri': [9, 9, 9, 9, 9, 8, 7, 7, 7, 4],
            'Lojistikte Sürdürülebilirlik': [10, 9, 9, 7, 7, 7, 7, 6, 6, 4]
        },
        'Amazon': {
            'Teslimat Süresi': [10, 10, 10, 10, 10, 10, 9, 9, 9, 2],
            'Müşteri Memnuniyeti': [10, 10, 10, 10, 9, 9, 9, 9, 9, 4],
            'İade Süreci Verimliliği': [10, 10, 10, 10, 10, 9, 9, 9, 8, 4],
            'Sipariş Doğruluğu': [9, 9, 9, 9, 9, 9, 10, 10, 3, 2],
            'Takip Sistemi Etkinliği': [10, 10, 10, 9, 9, 9, 9, 8, 7, 2],
            'Envanter Yönetimi Verimliliği': [10, 10, 10, 10, 9, 9, 9, 9, 8, 4],
            'Son Mil Teslimat Çözümleri': [10, 9, 9, 9, 9, 9, 9, 8, 7, 4],
            'Lojistikte Sürdürülebilirlik': [10, 10, 9, 9, 9, 9, 8, 8, 8, 4]
        },
        'Hepsiburada': {
            'Teslimat Süresi': [8, 8, 8, 8, 2, 4, 7, 7, 9, 10],
            'Müşteri Memnuniyeti': [8, 8, 8, 7, 7, 7, 4, 4, 9, 10],
            'İade Süreci Verimliliği': [9, 9, 9, 9, 7, 7, 7, 6, 4, 1],
            'Sipariş Doğruluğu': [10, 9, 9, 8, 8, 8, 7, 7, 4, 2],
            'Takip Sistemi Etkinliği': [10, 10, 8, 8, 7, 7, 7, 7, 1, 2],
            'Envanter Yönetimi Verimliliği': [10, 10, 8, 7, 7, 7, 7, 7, 4, 1],
            'Son Mil Teslimat Çözümleri': [8, 8, 8, 8, 8, 10, 9, 7, 6, 4],
            'Lojistikte Sürdürülebilirlik': [9, 9, 9, 8, 8, 6, 6, 6, 5, 4]
        }
    }
    # Lojistik Maliyeti için şirketlerin ham puanları (manuel eklenen)
    original_company_raw_scores['Trendyol']['Lojistik Maliyeti'] = [10,10,10,10,9,9,8,7,7,4]
    original_company_raw_scores['Amazon']['Lojistik Maliyeti'] = [10,10,10,9,9,8,8,7,6,3]
    original_company_raw_scores['Hepsiburada']['Lojistik Maliyeti'] = [10,10,8,8,8,7,7,6,3,2]

    # Ortalama performans değerlerini hesaplayıp initial_df'i dolduralım
    df_data = {}
    for crit in criteria:
        df_data[crit] = []
        for alt in alternatives:
            # Orijinal verilerde alternatif ve kriter eşleşiyorsa ortalama puanı al
            if alt in original_company_raw_scores and crit in original_company_raw_scores[alt]:
                df_data[crit].append(np.mean(original_company_raw_scores[alt][crit]))
            else:
                # Eşleşmiyorsa veya yeni bir alternatif/kriterse varsayılan 0.0
                df_data[crit].append(0.0)

    initial_df = pd.DataFrame(df_data, index=alternatives)

    # Karar matrisi sütun yapılandırması - Hassasiyeti artır
    column_configuration = {}
    for crit in criteria:
        column_configuration[crit] = st.column_config.NumberColumn(
            label=crit,
            format="%.6f",  # 6 ondalık hane hassasiyet
            min_value=0.0,
            max_value=1000.0 # Güncellendi: Artık 1000.0'e kadar değer girilebilir
        )

    edited_df = st.data_editor(
        initial_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=False,
        column_config=column_configuration # Yeni eklenen config
    )
    initial_decision_matrix = edited_df
else:
    st.warning("Lütfen önce alternatifleri ve kriterleri giriniz.")
    initial_decision_matrix = pd.DataFrame() # Boş DataFrame


# Zeta Değeri
st.subheader("Zeta (ζ) Değeri")
default_zeta = st.slider("Heron Ortalaması için Zeta (ζ) Değeri:", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

st.markdown("---")

# --- ARLON YÖNTEMİNİ ÇALIŞTIR ---
if st.button("Analizi Başlat") and not initial_decision_matrix.empty and not criterion_weights_series.empty:
    st.header("2. Analiz Sonuçları")

    try:
        # ARLON metodunu çalıştır
        st.subheader("ARLON Yöntemi Sonuçları")
        final_ranking_arlon, norm1_df, norm2_df, agg_norm_df, weighted_agg_norm_df, cost_benefit_sums = run_arlon_method_with_intermediate_results(
            initial_decision_matrix, criterion_weights_series, benefit_criteria, cost_criteria, default_zeta
        )

        st.subheader(f"Alternatiflerin Nihai ARLON Sıralaması (Zeta = {default_zeta})")
        st.dataframe(final_ranking_arlon.rename(columns={'Final_Ranking_Score (ℝ_i)': 'ARLON Skoru', 'Ranking': 'Sıralama'}), use_container_width=True)

        st.markdown("---")

        st.subheader("ARLON Ara Sonuçları")

        st.markdown("#### Logaritmik Normalizasyon Matrisi (İlk Aşama - $x_{ij}^{(1)}$)")
        st.markdown("Bu tablo, ARLON metodunun ilk logaritmik normalizasyon adımından sonra elde edilen $x_{ij}^{(1)}$ değerlerini göstermektedir (Eq. 9 ve Eq. 10).")
        st.dataframe(norm1_df, use_container_width=True)

        st.markdown("#### Logaritmik Normalizasyon Matrisi (İkinci Aşama - $x_{ij}^{(2)}$)")
        st.dataframe(norm2_df, use_container_width=True)

        st.markdown("#### Heron Ortalaması Agregasyon Matrisi ($x_{ij}^{(3)}$)")
        st.dataframe(agg_norm_df, use_container_width=True)

        st.markdown("#### Ağırlıklı Normalleştirilmiş Karar Matrisi")
        st.dataframe(weighted_agg_norm_df, use_container_width=True)

        st.markdown("#### Fayda ve Maliyet Toplamları ($B_i$ ve $C_i$)")
        st.dataframe(cost_benefit_sums, use_container_width=True)

        st.markdown("---")

        st.subheader("ARLON İçin Farklı Zeta Değerleri İçin Hassasiyet Analizi")
        zeta_values = np.linspace(0.0, 1.0, 11) # 0.0'dan 1.0'a 0.1 aralıklarla
        multi_zeta_results_df = pd.DataFrame(index=initial_decision_matrix.index)
        for zeta_val in zeta_values:
            zeta_results, _, _, _, _, _ = run_arlon_method_with_intermediate_results(initial_decision_matrix, criterion_weights_series, benefit_criteria, cost_criteria, zeta_val)
            multi_zeta_results_df[f'Zeta={zeta_val:.1f}'] = zeta_results['Final_Ranking_Score (ℝ_i)']

        st.dataframe(multi_zeta_results_df, use_container_width=True)

        st.subheader("ARLON: Zeta Değerlerine Göre Alternatif Skorlarının Değişimi")
        fig, ax = plt.subplots(figsize=(10, 6))
        multi_zeta_results_df.T.plot(ax=ax, marker='o')
        ax.set_title('Zeta Değerine Göre ARLON Skorlarının Değişimi')
        ax.set_xlabel('Zeta Değeri')
        ax.set_ylabel('ARLON Skoru')
        ax.grid(True)
        ax.legend(title='Alternatifler', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")

        # --- ARAS YÖNTEMİNİ ÇALIŞTIR ---
        st.subheader("ARAS Yöntemi Sonuçları")
        final_ranking_aras = run_aras_method(initial_decision_matrix, criterion_weights_series, benefit_criteria, cost_criteria)
        st.subheader("Alternatiflerin Nihai ARAS Sıralaması")
        st.dataframe(final_ranking_aras.rename(columns={'ARAS_Score (K_i)': 'ARAS Skoru', 'Ranking': 'Sıralama'}), use_container_width=True)

        st.markdown("---")

        # --- ARLON ve ARAS Yöntemi Sıralama Karşılaştırması ---
        st.header("3. Yöntemler Arası Sıralama Karşılaştırması")
        comparison_df = pd.DataFrame({
            'ARLON Sıralama': final_ranking_arlon['Ranking'],
            'ARAS Sıralama': final_ranking_aras['Ranking']
        })
        st.dataframe(comparison_df, use_container_width=True)

        st.subheader("Sıralama Farklılıkları")
        # Eğer sıralamalar tamamen aynıysa bilgilendirme mesajı
        if (comparison_df['ARLON Sıralama'] == comparison_df['ARAS Sıralama']).all():
            st.info("ARLON ve ARAS yöntemlerinin sıralamaları aynıdır.")
        else:
            st.warning("ARLON ve ARAS yöntemlerinin sıralamaları arasında farklılıklar bulunmaktadır.")


    except ValueError as e:
        st.error(f"Hesaplama Hatası: {e}")
    except Exception as e:
        st.error(f"Beklenmedik bir hata oluştu: {e}")
else:
    st.info("Lütfen tüm veri girişlerini tamamlayın ve 'Analizi Başlat' butonuna tıklayın.")
