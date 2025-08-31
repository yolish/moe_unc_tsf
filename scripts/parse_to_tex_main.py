import pandas as pd
import numpy as np



def to_value(a):
    assert len(a) <= 1
    if len(a) == 1:
        return a[0]
    else:
        return -1

df = pd.read_csv("moe_results.csv")
domains_to_names = {"weather":"Weather","electricity":"Electricity","traffic":"Traffic","national-illness":"ILI",
                    "ETTh1":"ETTh1","ETTh2":"ETTh2","ETTm1":"ETTm1","ETTm2":"ETTm2"}
experts = ["DLinear", "iTransformer", "PatchTST"]


print("")
domains = np.sort(list(domains_to_names.keys()))
for domain in domains:
    domain_name = domains_to_names[domain]
    if domain == "national-illness":
        pred_lens = [24, 36, 48, 60]
    else:
        pred_lens = [96, 192, 336, 720]
    for i, pl in enumerate(pred_lens):
        row = []
        if i == 0:
            row.append("&\\multirow{" + "4}*" + "{\\rotatebox{"+"90}{"+"{}".format(domain_name)+"}}")
        else:
            row.append("&\\multicolumn{"+"1}{c|}{}")
        row.append(pl)
        for e in experts:
            domain_df = df[(df["dataset"] == domain) & (df["expert"] == e) & (df["pl"] == pl)]
            
            det_moe = domain_df[(domain_df["num_experts"] == 3) & (domain_df["prob_moe"] == 0)]
            if det_moe.empty:
               det_moe_mse = -1 
            else:
                det_moe_mse = to_value(det_moe["mse"].values)
            
            #det_moe_mae = to_value(det_moe["mae"].values)
            prob_moe = domain_df[(domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 1)]
            if prob_moe.empty:
                prob_moe_mse = -1 
            else:
                prob_moe_mse = to_value(prob_moe["mse"].values)
            
            if det_moe_mse == -1 or prob_moe_mse == -1:
                if det_moe_mse == -1:
                    row.append("M")
                else:
                    row.append("{:.3f}".format(det_moe_mse))
                if prob_moe_mse == -1:
                    row.append("M")
                else:
                    row.append("{:.3f}".format(prob_moe_mse))
            else:
                if det_moe_mse < prob_moe_mse:
                    row.append("\\textbf{" + "{:.3f}".format(det_moe_mse) + "}")
                    row.append("{:.3f}".format(prob_moe_mse))
                elif det_moe_mse == prob_moe_mse:
                    row.append("\\textbf{" + "{:.3f}".format(det_moe_mse) + "}")
                    row.append("\\textbf{" + "{:.3f}".format(prob_moe_mse) + "}")
                else:
                    row.append("{:.3f}".format(det_moe_mse))
                    row.append("\\textbf{" + "{:.3f}".format(prob_moe_mse) + "}")
        print("{} & {} & {} & {} & {} & {} & {} & {} \\\\".format(*row))
    print("\\cline{2-9}")
            



    
"""
config = 3
expert = experts[1]
if config == 1:
    print("domain,pred_len,det_moe_mse,det_moe_mae,prob_moe_mse,prob_moe_mae")
    for d in domains:
        domain_df = df[(df["dataset"] == d) & (df["expert"] == expert)]
        pred_lens = np.sort(np.unique(domain_df["pl"].values))
        for pl in pred_lens:
            det_moe = domain_df[(domain_df["pl"] == pl) & \
                                (domain_df["num_experts"] == 3) & (domain_df["prob_moe"] == 0)]
            det_moe_mse = to_value(det_moe["mse"].values)
            det_moe_mae = to_value(det_moe["mae"].values)
            prob_moe = domain_df[(domain_df["pl"] == pl) & (domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 1)]
            prob_moe_mse = to_value(prob_moe["mse"].values)
            prob_moe_mae = to_value(prob_moe["mae"].values)
            print("{},{},{:.3f},{:.3f},{:.3f},{:.3f}".format(d, pl, det_moe_mse, det_moe_mae, prob_moe_mse, prob_moe_mae))
elif config == 2:
    print("domain,pred_len,det_mse,det_moe_mse,prob_moe_nu_mse,prob_moe_mse")
    for d in domains:
        domain_df = df[(df["dataset"] == d) & (df["expert"] == expert)]
        pred_lens = np.sort(np.unique(domain_df["pl"].values))
        for pl in pred_lens:
            det = domain_df[(domain_df["pl"] == pl) & \
                                (domain_df["num_experts"] == 1) & (domain_df["prob_moe"] == 0)]
            det_mse = to_value(det["mse"].values)

            det_moe = domain_df[(domain_df["pl"] == pl) & \
                                (domain_df["num_experts"] == 3) & (domain_df["prob_moe"] == 0)]
            det_moe_mse = to_value(det_moe["mse"].values)

            prob_moe = domain_df[(domain_df["pl"] == pl) & (domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 1)]
            prob_moe_mse = to_value(prob_moe["mse"].values)

            prob_moe_nu = domain_df[(domain_df["pl"] == pl) & (domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 0)]
            prob_moe_nu_mse = to_value(prob_moe_nu["mse"].values)
            print("{},{},{:.3f},{:.3f},{:.3f},{:.3f}".format(d, pl, det_mse, det_moe_mse, prob_moe_nu_mse, prob_moe_mse))
elif config == 3:
    print("domain,pred_len,det_moe_mse,prob_moe_nu_mse,prob_moe_mse")
    for d in domains:
        domain_df = df[(df["dataset"] == d) & (df["expert"] == expert)]
        pred_lens = np.sort(np.unique(domain_df["pl"].values))
        for pl in pred_lens:
            det_moe = domain_df[(domain_df["pl"] == pl) & \
                                (domain_df["num_experts"] == 3) & (domain_df["prob_moe"] == 0)]
            det_moe_mse = to_value(det_moe["mse"].values)

            prob_moe = domain_df[(domain_df["pl"] == pl) & (domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 1)]
            prob_moe_mse = to_value(prob_moe["mse"].values)

            prob_moe_nu = domain_df[(domain_df["pl"] == pl) & (domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 0)]
            prob_moe_nu_mse = to_value(prob_moe_nu["mse"].values)
            print("{},{},{:.3f},{:.3f},{:.3f}".format(d, pl, det_moe_mse, prob_moe_nu_mse, prob_moe_mse))
"""
