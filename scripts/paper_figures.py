"""Paper figures: matplotlib publication-quality plots"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import os

OUTPUT_DIR = "output/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def load_data():
    mp = pd.read_csv("output/master_panel_data_v2.csv", encoding='utf-8-sig')
    ps = pd.read_csv("output/policy_scores_panel.csv", encoding='utf-8-sig')
    scm = pd.read_csv("analysis/scm_results.csv", encoding='utf-8-sig')
    ti = pd.read_csv("output/textile_indices_annual.csv", encoding='utf-8-sig')
    return mp, ps, scm, ti


def fig1_timeseries(mp, ps):
    """Figure 1: Textile firms + Policy intensity dual-axis time series"""
    gy_ps = ps[ps['County_Code'].astype(str) == '130628']
    m = mp[['Year', 'textile_firms']].merge(gy_ps[['Year', 'policy_intensity_total']], on='Year')

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_enterprise = '#2c3e50'
    color_policy = '#e74c3c'

    ax1.bar(m['Year'], m['textile_firms'], color=color_enterprise, alpha=0.85, width=0.7)
    ax1.set_ylabel('Textile Firms (Annual Registration)', color=color_enterprise)
    ax1.tick_params(axis='y', labelcolor=color_enterprise)
    ax1.set_ylim(0, max(m['textile_firms']) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(m['Year'], m['policy_intensity_total'], color=color_policy, linewidth=2.5,
             marker='o', markersize=6, markerfacecolor=color_policy)
    ax2.fill_between(m['Year'], 0, m['policy_intensity_total'], alpha=0.1, color=color_policy)
    ax2.set_ylabel('LLM Policy Intensity Score', color=color_policy)
    ax2.tick_params(axis='y', labelcolor=color_policy)
    ax2.set_ylim(0, max(m['policy_intensity_total']) * 1.2)

    ax1.axvline(x=2017, color='gray', linestyle='--', alpha=0.6, linewidth=1.2)
    ax1.text(2017.2, ax1.get_ylim()[1] * 0.95, '2017 Environmental\nInspection', fontsize=9, color='gray')

    ax1.set_xlabel('Year')
    ax1.set_xlim(1999.5, 2026.5)
    fig.suptitle('Gaoyang County: Textile Firm Registrations and LLM Policy Intensity (2000-2026)',
                 fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig1_timeseries.png")
    plt.close()
    print("  Fig1 saved")


def fig2_scm_gap(scm):
    """Figure 2: SCM - Gaoyang vs Synthetic Gaoyang"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7))

    # Top: trajectory comparison
    pre = scm[scm['Period'] == 'Pre']
    post = scm[scm['Period'] == 'Post']

    ax1.plot(pre['Year'], pre['Gaoyang_actual'], 'b-', linewidth=2, label='Gaoyang (actual)')
    ax1.plot(pre['Year'], pre['Synthetic_Gaoyang'], 'r--', linewidth=2, label='Synthetic Gaoyang')
    ax1.plot(post['Year'], post['Gaoyang_actual'], 'b-', linewidth=2)
    ax1.plot(post['Year'], post['Synthetic_Gaoyang'], 'r--', linewidth=2)
    ax1.axvline(x=2017, color='gray', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Textile Firms')
    ax1.legend(loc='upper left')
    ax1.text(2017.2, ax1.get_ylim()[1] * 0.95, '2017', fontsize=9, color='gray')

    # Bottom: gap plot
    gaps = list(pre['Gap']) + list(post['Gap'])
    years = list(scm['Year'])
    colors = ['#3498db' if g >= 0 else '#e74c3c' for g in gaps]
    ax2.bar(years, gaps, color=colors, alpha=0.8, width=0.7)
    ax2.axhline(y=0, color='black', linewidth=0.5)
    ax2.axvline(x=2017, color='gray', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Gap (Actual - Synthetic)')
    ax2.set_xlabel('Year')

    fig.suptitle('Synthetic Control: Gaoyang vs Synthetic Counterpart (Li County)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig2_scm.png")
    plt.close()
    print("  Fig2 saved")


def fig3_model_comparison(mp, ps):
    """Figure 3: Coefficient plot - policy dimensions with and without time control"""
    gy_ps = ps[ps['County_Code'].astype(str) == '130628']
    m = mp[['Year', 'textile_firms']].merge(gy_ps[['Year', 'policy_intensity_total',
        'equipment_index', 'environment_index', 'ecommerce_index',
        'brandquality_index', 'cluster_index', 'finance_index', 'education_index']], on='Year')
    m = m[m['Year'] <= 2024]
    m['time'] = m['Year'] - 2000

    dims = ['policy_intensity_total', 'equipment_index', 'environment_index',
            'ecommerce_index', 'brandquality_index', 'cluster_index', 'finance_index', 'education_index']
    labels = ['Total Policy', 'Equipment', 'Environment', 'Ecommerce',
              'Brand Quality', 'Cluster', 'Finance', 'Education (placebo)']

    results_no_time = []
    results_with_time = []

    for dim in dims:
        X = np.column_stack([np.ones(len(m)), m[dim].values])
        b = np.linalg.inv(X.T @ X) @ X.T @ m['textile_firms'].values
        r = m['textile_firms'].values - X @ b
        n = len(m)
        se = np.sqrt(np.sum(r**2) / (n-2) * np.diag(np.linalg.inv(X.T @ X)))
        results_no_time.append({'dim': dim, 'coef': b[1], 'se': se[1],
                                'ci_low': b[1] - 1.96*se[1], 'ci_high': b[1] + 1.96*se[1]})

        X2 = np.column_stack([np.ones(len(m)), m[dim].values, m['time'].values])
        b2 = np.linalg.inv(X2.T @ X2) @ X2.T @ m['textile_firms'].values
        r2 = m['textile_firms'].values - X2 @ b2
        se2 = np.sqrt(np.sum(r2**2) / (n-3) * np.diag(np.linalg.inv(X2.T @ X2)))
        results_with_time.append({'dim': dim, 'coef': b2[1], 'se': se2[1],
                                   'ci_low': b2[1] - 1.96*se2[1], 'ci_high': b2[1] + 1.96*se2[1]})

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, title, results in [
        (ax1, 'Without Time Control\n(Spurious)', results_no_time),
        (ax2, 'With Time Control\n(Correct Specification)', results_with_time)
    ]:
        y_pos = range(len(results))
        coefs = [r['coef'] for r in results]
        errors = [[r['coef'] - r['ci_low'] for r in results],
                   [r['ci_high'] - r['coef'] for r in results]]
        colors = ['#2c3e50' if r['ci_low'] > 0 else '#bdc3c7' for r in results]
        ax.barh(y_pos, coefs, color=colors, alpha=0.85, height=0.6)
        ax.errorbar(coefs, y_pos, xerr=errors, fmt='none', ecolor='black', capsize=3)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Coefficient (firms per unit)')
        ax.set_title(title)

    fig.suptitle('Policy Dimension Effects: The Danger of Omitted Time Trend', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig3_model_comparison.png")
    plt.close()
    print("  Fig3 saved")


def fig4_correlation_heatmap(mp, ps):
    """Figure 4: Correlation heatmap"""
    gy_ps = ps[ps['County_Code'].astype(str) == '130628']
    m = mp[['Year', 'textile_firms', 'total_firms']].merge(gy_ps[['Year', 'policy_intensity_total',
        'equipment_index', 'environment_index', 'ecommerce_index',
        'brandquality_index', 'cluster_index', 'finance_index', 'education_index']], on='Year')
    m['time'] = m['Year'] - 2000

    cols = ['textile_firms', 'total_firms', 'policy_intensity_total',
            'equipment_index', 'environment_index', 'ecommerce_index',
            'brandquality_index', 'cluster_index', 'finance_index', 'education_index', 'time']
    labels = ['Textile Firms', 'Total Firms', 'Policy Total', 'Equipment', 'Environment',
              'Ecommerce', 'Brand Quality', 'Cluster', 'Finance', 'Education', 'Time Trend']

    corr = m[cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, square=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'shrink': 0.8, 'label': 'Pearson r'},
                ax=ax)
    ax.set_title('Correlation Matrix: Key Variables', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig4_correlation_heatmap.png")
    plt.close()
    print("  Fig4 saved")


def fig5_construct_validity(ti, ps):
    """Figure 5: Construct validity - Official vs LLM policy scores"""
    gy_ps = ps[ps['County_Code'].astype(str) == '130628']
    m = ti[['Year', 'index_policy_support']].merge(
        gy_ps[['Year', 'policy_intensity_total']], on='Year', how='inner')

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(m['index_policy_support'], m['policy_intensity_total'],
               s=80, c=m['Year'], cmap='viridis', edgecolors='black', linewidth=0.5)

    corr = m['index_policy_support'].corr(m['policy_intensity_total'])
    ax.text(0.05, 0.95, f'r = {corr:.3f} (n={len(m)})', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    for _, row in m.iterrows():
        ax.annotate(str(int(row['Year'])), (row['index_policy_support'], row['policy_intensity_total']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)

    ax.set_xlabel('Official Policy Support Index (Textile Association)')
    ax.set_ylabel('LLM Policy Intensity Score')
    ax.set_title('Construct Validity: Official Index vs LLM Score (2020-2026)', fontweight='bold')
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig5_construct_validity.png")
    plt.close()
    print("  Fig5 saved")


def fig6_policy_heatmap(mp, ps):
    """Figure 6: Policy dimension heatmap by year"""
    gy_ps = ps[ps['County_Code'].astype(str) == '130628']
    dims = ['equipment_index', 'environment_index', 'ecommerce_index',
            'brandquality_index', 'cluster_index', 'finance_index', 'education_index']
    labels = ['Equipment', 'Environment', 'Ecommerce', 'Brand', 'Cluster', 'Finance', 'Education']

    heatmap_data = gy_ps[['Year'] + dims].set_index('Year').T
    heatmap_data = heatmap_data.loc[:, heatmap_data.columns >= 2009]

    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd',
                linewidths=0.5, cbar_kws={'label': 'Policy Score', 'shrink': 0.8},
                xticklabels=True, yticklabels=labels, ax=ax)
    ax.set_title('Policy Dimension Scores by Year (Gaoyang County)', fontweight='bold')
    ax.set_xlabel('Year')
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig6_policy_heatmap.png")
    plt.close()
    print("  Fig6 saved")


def fig7_textile_subindustry(mp):
    """Figure 7: Textile industry breakdown over time"""
    ti_sub = pd.read_csv("output/gaoyang_textile_registration.csv", encoding='utf-8-sig')
    pivot = ti_sub.pivot_table(index='Year', columns='industry_level2',
                                values='firm_count', aggfunc='sum').fillna(0)
    pivot_2000 = pivot[pivot.index >= 2000].astype(int)

    fig, ax = plt.subplots(figsize=(10, 5))
    industries = pivot_2000.columns.tolist()
    colors = ['#2c3e50', '#e74c3c']

    bottom = np.zeros(len(pivot_2000))
    for i, ind in enumerate(industries):
        ax.bar(pivot_2000.index, pivot_2000[ind], bottom=bottom,
               color=colors[i], alpha=0.85, label=ind, width=0.8)
        bottom += pivot_2000[ind].values

    ax.axvline(x=2017, color='gray', linestyle='--', linewidth=1.2)
    ax.text(2017.2, ax.get_ylim()[1] * 0.95, '2017', fontsize=10, color='gray')
    ax.set_xlabel('Year')
    ax.set_ylabel('Firm Registrations')
    ax.set_title('Textile Industry Registrations by Sub-category (Gaoyang)', fontweight='bold')
    ax.legend(loc='upper left')
    ax.set_xlim(1999.5, 2024.5)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig7_textile_subindustry.png")
    plt.close()
    print("  Fig7 saved")


def fig8_descriptive_pre_post(mp, ps):
    """Figure 8: Pre vs Post 2017 comparison"""
    gy_ps = ps[ps['County_Code'].astype(str) == '130628']
    m = mp[['Year', 'textile_firms', 'total_firms']].merge(
        gy_ps[['Year', 'policy_intensity_total']], on='Year')
    m['period'] = m['Year'].apply(lambda y: 'Pre-2017' if y < 2017 else 'Post-2017')

    var_labels = {
        'textile_firms': 'Textile Firms',
        'total_firms': 'Total Firms',
        'policy_intensity_total': 'Policy Intensity\n(LLM Score)'
    }

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5))
    for ax, (var, label) in zip(axes, var_labels.items()):
        pre_data = m[m['period'] == 'Pre-2017'][var].dropna()
        post_data = m[m['period'] == 'Post-2017'][var].dropna()

        bp = ax.boxplot([pre_data, post_data], labels=['Pre-2017\n(2000-2016)', 'Post-2017\n(2017-2026)'],
                         patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#3498db')
        bp['boxes'][1].set_facecolor('#e74c3c')

        for i, data in enumerate([pre_data, post_data]):
            jitter = np.random.normal(i + 1, 0.04, size=len(data))
            ax.scatter(jitter, data, alpha=0.4, s=20, color='black')

        ax.set_ylabel(label)
        ax.set_title(label)

    fig.suptitle('Pre vs Post 2017: Key Variables', fontweight='bold', fontsize=14)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fig8_pre_post_boxplot.png")
    plt.close()
    print("  Fig8 saved")


def main():
    print("Generating paper figures...")
    mp, ps, scm, ti = load_data()

    fig1_timeseries(mp, ps)
    fig2_scm_gap(scm)
    fig3_model_comparison(mp, ps)
    fig4_correlation_heatmap(mp, ps)
    fig5_construct_validity(ti, ps)
    fig6_policy_heatmap(mp, ps)
    fig7_textile_subindustry(mp)
    fig8_descriptive_pre_post(mp, ps)

    print(f"\nAll 8 figures saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
