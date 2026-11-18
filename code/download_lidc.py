"""Usage: python download_lidc.py N [OFFSET]. Download CT scans of N LIDC-IDRI patients (one CT series per patient) via idc-index."""
import sys
from pathlib import Path
from idc_index import IDCClient

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
OFFSET = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # skip the first OFFSET patients
OUT = Path(__file__).resolve().parent.parent / "dataset" / "LIDC-IDRI"
OUT.mkdir(parents=True, exist_ok=True)

c = IDCClient()
df = c.sql_query(f"""
    SELECT PatientID, ANY_VALUE(SeriesInstanceUID) AS SeriesInstanceUID,
           ANY_VALUE(instanceCount) AS slices, ANY_VALUE(series_size_MB) AS mb
    FROM index WHERE collection_id='lidc_idri' AND Modality='CT'
    GROUP BY PatientID ORDER BY PatientID LIMIT {N} OFFSET {OFFSET}
""")
sel = OUT.parent / "selected_patients.csv"  # cumulative list of all patients ever selected
if sel.exists():
    import pandas as pd
    df_all = pd.concat([pd.read_csv(sel), df]).drop_duplicates("PatientID")
else:
    df_all = df
df_all.to_csv(sel, index=False)
print(f"{len(df)} patients, {int(df.slices.sum())} slices, {df.mb.sum()/1024:.1f} GB")
c.download_from_selection(seriesInstanceUID=list(df.SeriesInstanceUID), downloadDir=str(OUT),
                          dirTemplate="%PatientID/%SeriesInstanceUID")
print("Done:", OUT)
