from scipy.stats import wilcoxon
import numpy as np
import json
# from statsmodels.stats.multitest import multipletests



def paired_wilcoxon(model_scores, random_scores, alternative="greater"):
    model_scores = np.asarray(model_scores, dtype=float)
    random_scores = np.asarray(random_scores, dtype=float)

    if model_scores.shape != random_scores.shape:
        raise ValueError("model_scores and random_scores must have same shape")

    diff = model_scores - random_scores

    stat, p = wilcoxon(
        model_scores,
        random_scores,
        alternative=alternative,  # "greater" if testing model > random
        zero_method="wilcox",
    )

    return {
        "n_pairs": len(diff),
        "mean_model": float(model_scores.mean()),
        "mean_random": float(random_scores.mean()),
        "mean_delta": float(diff.mean()),
        "median_delta": float(np.median(diff)),
        "wilcoxon_stat": float(stat),
        "p_value": float(p),
    }
    
    
def paired_effect_size(model_scores, random_scores):
    diff = np.asarray(model_scores) - np.asarray(random_scores)
    return float(diff.mean() / diff.std(ddof=1))


if __name__ == "__main__":
    
    # 10 cell types
    # prauc_model_preds = [0.117980337, 0.148079096, 0.503294866, 0.215328212, 0.284291536, 0.513979079, 0.476767444, 0.940647384, 0.542936632, 0.183677838]
    # prauc_random_preds = [0.04211603, 0.03833685, 0.16476569, 0.04676441, 0.06748514, 0.07183844,
    #                       0.16270941, 0.1942189,  0.19106459, 0.02070054]
    
    # f1_model_preds = [0.00333, 0.1099, 0.5315, 0.1103, 0.3521, 0.4271, 0.4154, 0.8630, 0.4853, 0.0069]
    # f1_random_preds = [0.03956178639506208, 0.050744109356905404, 0.2030631537779395, 0.0378247203719037, 0.07207235464605792, 0.07314186136277187, 0.13080401675816783, 0.2031141629331471, 0.1780799933669056, 0.009926518129265207]
    
    # acc_model_preds = [0.001744847354292734, 0.06773264712634883, 0.6966013952001153, 0.06901397963552428, 0.37781013973056066, 0.3735248040085194, 0.3621078074617663, 0.8629908173016995, 0.5718202365737127, 0.0035267729311825903]
    # acc_random_preds = [0.045884434092034555, 0.049864336803648455, 0.20116410273159793, 0.03854768261380903, 0.07457393093045049, 0.07223302542205082, 0.13595143128201886, 0.19277417275302386, 0.17914964878786788, 0.009877702443392606]
    
    # 5 cell types
    # prauc_model_preds = [0.940647384, 0.73866939, 0.419002214, 0.76399035, 0.542936632]
    # prauc_random_preds = [0.1942189, 0.29198298, 0.08818569, 0.23454785, 0.19106459]
    
    # f1_model_preds = [0.8631, 0.6746, 0.3918, 0.6253, 0.4771]
    # f1_random_preds = [0.2031, 0.3320, 0.0820, 0.2041, 0.1781]
    
    acc_model_preds = [0.854138031329206, 0.7617518117621256, 0.32372585748950683, 0.561166949899581, 0.48793328243881934]
    acc_random_preds = [0.19277951567599041, 0.33546649743385215, 0.08444162970487896, 0.2081745430557702, 0.17915468643060406]
    
    type_test = "accuracy"  # "f1" or "prauc" or "accuracy"
    # pw_test_results = paired_wilcoxon(prauc_model_preds, prauc_random_preds, alternative="greater")
    pw_test_results = paired_wilcoxon(acc_model_preds, acc_random_preds, alternative="greater")

    
    output_path = "/taiga/illinois/vetmed/cb/kwang222/cellsighter_testing/shirui_code/CellSighter/src/data/figure_images/results"
    
    # Save results to a  file
    with open(f"{output_path}/{type_test}_wilcoxon_test_results_5ct.json", "w") as f:
        json.dump(pw_test_results, f, indent=4)
    
    