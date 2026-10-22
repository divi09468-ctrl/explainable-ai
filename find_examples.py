import pandas as pd
from xai_module import model

columns = [
    "duration", "protocol_type", "service", "flag", "src_bytes",
    "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

df = pd.read_csv(
    "dataset/KDDTest+.txt",
    header=None,
    names=columns
)

features = df.drop(columns=["label", "difficulty"])
predictions = model.predict(features)

for i, prediction in enumerate(predictions):
    if str(prediction).lower() in ["normal", "0"]:
        print("First predicted Normal record number:", i + 1)
        print("Actual dataset label:", df.iloc[i]["label"])
        break
else:
    print("No Normal prediction found.")