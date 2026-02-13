import os
import csv
import argparse
import matplotlib.pyplot as plt

def read_metrics(csv_path):
    epochs, train_losses, dev_losses, bleus = [], [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            dev_losses.append(float(row["dev_loss"]))
            bleus.append(float(row["bleu"]))
    return epochs, train_losses, dev_losses, bleus

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/metrics.csv")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    csv_path = args.csv
    outdir = args.outdir or os.path.dirname(csv_path)
    os.makedirs(outdir, exist_ok=True)

    epochs, train_losses, dev_losses, bleus = read_metrics(csv_path)

    # loss 图
    plt.figure()
    plt.plot(epochs, train_losses, label="train_loss")
    plt.plot(epochs, dev_losses, label="dev_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_loss.png"), dpi=200)
    plt.close()

    # BLEU 图
    plt.figure()
    plt.plot(epochs, bleus, label="BLEU")
    plt.xlabel("epoch")
    plt.ylabel("BLEU")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "curve_bleu.png"), dpi=200)
    plt.close()

    print("saved:", os.path.join(outdir, "curve_loss.png"))
    print("saved:", os.path.join(outdir, "curve_bleu.png"))

if __name__ == "__main__":
    main()
