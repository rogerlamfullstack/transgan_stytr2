import re
from collections import defaultdict

# Put your log files here
log_files = [
    "test_single_gan_20000_loss.log",
    "test_single_gan_25000_loss.log",
    "test_single_gan_40000_loss.log"
    # Add more if needed
]

# Regex patterns
pattern1 = re.compile(
    r"loss_content=(?P<loss_content>[\d.]+),\s*"
    r"loss_style=(?P<loss_style>[\d.]+),\s*"
    r"loss_identity1=(?P<loss_identity1>[\d.]+),\s*"
    r"loss_identity2=(?P<loss_identity2>[\d.]+),\s*"
    r"loss_contrastive_c=(?P<loss_contrastive_c>[\d.]+),\s*"
    r"loss_contrastive_s=(?P<loss_contrastive_s>[\d.]+),\s*"
    r"loss_gan_g=(?P<loss_gan_g>[\d.]+),\s*"
    r"loss_gan_d=(?P<loss_gan_d>[\d.]+),\s*"
    r"total_loss=(?P<total_loss>[\d.]+)"
)

pattern2 = re.compile(
    r"total_loss:\s*(?P<total_loss>[\d.]+),\s*losses:\s*"
    r"content:\s*(?P<content>[\d.]+)\s*\|\s*"
    r"global:\s*(?P<global_loss>[\d.]+)\s*\|\s*"
    r"local:\s*(?P<local_loss>[\d.]+)"
)

pattern3 = re.compile(
    r"content:\s*(?P<content>[\d.]+)\s*-\s*"
    r"style:\s*(?P<style>[\d.]+)\s*-\s*"
    r"l1:\s*(?P<l1>[\d.]+)\s*-\s*"
    r"l2:\s*(?P<l2>[\d.]+)"
)

pattern4 = re.compile(
    r"([0-9.]+)\s+-content:\s*([0-9.]+)\s+-style:\s*([0-9.]+)\s+-l1:\s*([0-9.eE+-]+)\s+-l2:\s*([0-9.]+)"
)

# Process each file
for log_file in log_files:
    loss_sums = defaultdict(float)
    count = 0

    try:
        with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                match1 = pattern1.search(line)
                match2 = pattern2.search(line)
                match3 = pattern3.search(line)
                match4 = pattern4.search(line)

                # Handle pattern4 separately since it has no named groups
                if match4:
                    groups = match4.groups()
                    keys = ["loss", "content", "style", "l1", "l2"]
                    match_dict = dict(zip(keys, groups))
                else:
                    match = match1 or match2 or match3
                    match_dict = match.groupdict() if match else None

                if match_dict:
                    count += 1
                    for key, value in match_dict.items():
                        try:
                            loss_sums[key] += float(value)
                        except ValueError:
                            pass
    except FileNotFoundError:
        print(f"File not found: {log_file}")
        continue

    # Print result
    print(f"File: {log_file}")
    if count > 0:
        print(f"Processed {count} log entries\n")
        for key in sorted(loss_sums.keys()):
            avg_loss = loss_sums[key] / count
            print(f"{key}: {avg_loss:.6f}")
    else:
        print("No matching log entries found.")
