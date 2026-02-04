import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_size_suitability(length):
    """Gaussian-like penalty centered at 80-200 bp (optimal TLR9/cGAS sensing)."""
    if 80 <= length <= 200:
        return 1.0
    elif length < 80:
        return np.exp(-0.5 * ((length - 80) / 20)**2)
    else:
        return np.exp(-0.5 * ((length - 200) / 50)**2)

def calculate_repeat_density(sequence):
    """Fraction of bases in homopolymers >= 3 (Structural instability proxy)."""
    if not sequence: return 0.0
    count, i, n = 0, 0, len(sequence)
    while i < n:
        j = i
        while j < n and sequence[j] == sequence[i]: j += 1
        run_length = j - i
        if run_length >= 3: count += run_length
        i = j
    return count / n

def analyze_mtdna_frp(sequence, dloop_coords=(16024, 576)):
    """
    FRP Analysis Tool: Fragmentation-Recognition Paradigm mapping.
    Handles circular DNA wrapping and calculates D-loop enrichment.
    """
    seq_len = len(sequence)
    extended_seq = sequence + sequence  # Circular handling
    fragments = []
    d_start, d_end = dloop_coords

    # Step 1: Circular-Aware Fragmentation
    for flen in range(50, 310, 10):
        for start in range(0, seq_len, 10):
            end = start + flen
            frag_seq = extended_seq[start:end].upper()
            
            # Feature Computation
            at_content = (frag_seq.count('A') + frag_seq.count('T')) / flen
            instability = 1.0 - ((frag_seq.count('G') + frag_seq.count('C')) / flen)
            repeat_density = calculate_repeat_density(frag_seq)
            
            # Check overlap with circular D-loop
            is_dloop = False
            for pos in range(start, end):
                actual_pos = pos % seq_len
                if d_start < d_end:
                    if d_start <= actual_pos <= d_end: is_dloop = True
                else: # Wraps around junction
                    if actual_pos >= d_start or actual_pos <= d_end: is_dloop = True
            
            cpg_density = frag_seq.count('CG') / flen
            size_suit = calculate_size_suitability(flen)
            
            fragments.append({
                'start': start,
                'end': end % seq_len,
                'length': flen,
                'is_dloop': is_dloop,
                'raw_fps': (0.3*at_content + 0.3*instability + 0.2*repeat_density + 0.2*float(is_dloop)),
                'raw_ivs': (0.6*cpg_density + 0.4*size_suit)
            })

    df = pd.DataFrame(fragments)
    eps = 1e-9

    # Step 2: Critical Normalization (Independent FPS/IVS scaling)
    df['FPS'] = (df['raw_fps'] - df['raw_fps'].min()) / (df['raw_fps'].max() - df['raw_fps'].min() + eps)
    df['IVS'] = (df['raw_ivs'] - df['raw_ivs'].min()) / (df['raw_ivs'].max() - df['raw_ivs'].min() + eps)
    
    # Combined Score and Global Percentile
    df['Final_score'] = df['FPS'] + df['IVS']
    df['Final_score_norm'] = (df['Final_score'] - df['Final_score'].min()) / (df['Final_score'].max() - df['Final_score'].min() + eps)
    df['percentile_rank'] = df['Final_score_norm'].rank(pct=True)

    # Step 3: Top-Fragment Extraction
    top_50 = df.sort_values('Final_score_norm', ascending=False).head(50)

    # Step 4: D-loop Enrichment Test (The "Alarm" Statistic)
    top_10pct = df[df['percentile_rank'] >= 0.9]
    dloop_baseline = df['is_dloop'].mean()
    enrichment = (top_10pct['is_dloop'].mean() / dloop_baseline) if dloop_baseline > 0 else 0

    return df, top_50, enrichment

def visualize_alarm_landscape(df, seq_len, dloop_coords, save_path="immune_alarm_plot.png"):
    """Plots the latent immunogenicity landscape of the mtDNA ring."""
    base_scores = np.zeros(seq_len)
    for _, row in df.iterrows():
        s, e, score = int(row['start']), int(row['end']), row['Final_score_norm']
        if s < e:
            base_scores[s:e] = np.maximum(base_scores[s:e], score)
        else: # Wrap around
            base_scores[s:] = np.maximum(base_scores[s:], score)
            base_scores[:e] = np.maximum(base_scores[:e], score)
    
    plt.figure(figsize=(14, 6))
    plt.plot(range(seq_len), base_scores, color='crimson', lw=1, label='Immunogenic Risk')
    plt.fill_between(range(seq_len), base_scores, color='crimson', alpha=0.15)
    
    # Highlight D-loop
    ds, de = dloop_coords
    if ds > de: # Circular wrap
        plt.axvspan(ds, seq_len, color='gray', alpha=0.2, label='D-loop (Control Region)')
        plt.axvspan(0, de, color='gray', alpha=0.2)
    else:
        plt.axvspan(ds, de, color='gray', alpha=0.2, label='D-loop (Control Region)')

    plt.title("The Latent Immune Alarm: mtDNA Immunogenicity Landscape (FRP Model)", fontsize=14)
    plt.xlabel("Genomic Position (bp)")
    plt.ylabel("Normalized Risk (Fragmentation x Recognition)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

# --- Execution ---
# sequence = "..." (Paste rCRS sequence here)
# df, top_50, enrichment = analyze_mtdna_frp(sequence)
# visualize_alarm_landscape(df, len(sequence), (16024, 576))
# print(f"D-loop Enrichment Factor: {enrichment:.2f}")
