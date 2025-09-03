import pandas as pd
import numpy as np



def to_value(a):
    assert len(a) <= 1
    if len(a) == 1:
        return a[0]
    else:
        return -1

df = pd.read_csv("moe_results.csv")
domains_to_names = {"ETTh1":"ETTh1","ETTh2":"ETTh2","ETTm1":"ETTm1","ETTm2":"ETTm2"}
experts = ["iTransformer", "PatchTST"]



print("")
domains = np.sort(list(domains_to_names.keys()))
pl = 96
for e in experts:
    
    for domain in domains:
        domain_name = domains_to_names[domain]
        row = []

        row.append(e)
        row.append(domain_name)

        domain_df = df[(df["dataset"] == domain) & (df["expert"] == e) & (df["pl"] == pl)]
            
        prob_moe_stan = domain_df[(domain_df["num_experts"] == 3) & \
                                (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 0)]
        
        row.append(to_value(prob_moe_stan["mae"].values))
        row.append(to_value(prob_moe_stan["mse"].values))
        prob_moe = domain_df[(domain_df["num_experts"] == 3) & \
                            (domain_df["prob_moe"] == 1) & (domain_df["unc_gating"] == 1)]
        row.append(to_value(prob_moe["mae"].values))
        row.append(to_value(prob_moe["mse"].values))
            
           
        print("{} & {} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\".format(*row))
        #print("{} & {} & {} & {} & {} & {} \\\\".format(*row))
    #print("\\cline{2-7}")
            



    
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
