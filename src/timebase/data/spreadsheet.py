import re

import pandas as pd

from timebase.data.static import *


def read(args) -> pd.DataFrame:
    """
    Loads clinical data spreadsheet and reshapes it to long format keeping
    only variable of interest
    """
    filename = os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.xlsx")
    if (not os.path.exists(filename)) or (args.overwrite):
        data = pd.read_excel(
            os.path.join(FILE_DIRECTORY, "TIMEBASE_database.xlsx"), sheet_name=0
        )
        states_n = list(data["N"])

        new_n = []
        for s, n in zip(
            STATES.keys(),
            np.diff([states_n.index(s) for s in STATES.values()] + [len(states_n)]),
        ):
            new_n.extend([s] * n)
        data["N"] = new_n
        data = data[~data["Control/Patient"].isnull()].reset_index(drop=True)
        col_selection = ["NHC", "age", "sex", "Session_Code", "YMRS", "HDRS", "IPAQ"]
        df_collector = []
        start = 0
        for status in STATES.keys():
            df = data[data["N"] == status]
            cols = [
                col for col in df.columns if any(map(col.__contains__, col_selection))
            ]
            df = df[cols]
            if len(df.loc[df["NHC"] == "-", "NHC"]):
                mock_ids = np.arange(
                    start, len(df.loc[df["NHC"] == "-", "NHC"]) + start
                )
                df.loc[df["NHC"] == "-", "NHC"] = mock_ids
                start += len(mock_ids)
            # df = df[df["age"].notnull()]
            df.columns = [
                col[3:] + "_" + col[:2] if bool(re.search("^T[0-9]_", col)) else col
                for col in df.columns
            ]
            df.insert(loc=3, column="status", value=[status] * len(df))
            df.drop_duplicates(subset="NHC", keep="first", inplace=True)
            stubnames = list(
                dict.fromkeys(
                    [col[:-3] for col in df.columns if bool(re.search("_T[0-9]$", col))]
                )
            )
            df = pd.wide_to_long(
                df,
                stubnames=stubnames,
                i=["NHC", "age", "sex", "status"],
                j="time",
                sep="_",
                suffix="\\w+",
            ).reset_index()
            df_collector.append(df)
        df = pd.concat(df_collector).dropna(axis=0).reset_index(drop=True)

        idx2keep = {}
        for i, c in enumerate(
            [
                re.findall(r"\d+", str(session_code))
                for session_code in df["Session_Code"]
            ]
        ):
            if len(c) == 0:
                idx2keep[i] = np.nan
            elif len(c) == 1:
                idx2keep[i] = c[0]
            else:
                idx2keep[i] = "\n".join([rec for rec in c if len(rec) >= 6])
        df["Session_Code"] = list(idx2keep.values())
        df = df.dropna(axis=0).reset_index(drop=True)

        # add duplicate row with unique Session_Code for sessions made up of multiple recordings
        rows = []
        idx = [
            i for i, v in enumerate(df["Session_Code"]) if bool(re.search("\n", str(v)))
        ]
        for i in idx:
            for subrec_id in df["Session_Code"].loc[i].split("\n"):
                row = df.loc[i].copy()
                row["Session_Code"] = subrec_id
                rows.append(
                    pd.DataFrame(
                        columns=list(df.columns),
                        data=row.values.reshape(1, len(row.values)),
                    )
                )
        df.drop(index=idx, axis=0, inplace=True)
        df = pd.concat(
            [pd.concat(rows).reset_index(drop=True), df], axis=0
        ).reset_index(drop=True)

        # male: 0, female:1
        df["sex"] = [1 if bool(re.search("^[Ff]", entry)) else 0 for entry in df.sex]
        df = df.astype(
            {
                col: int
                for col in df.columns
                if col not in ["status", "time", "Session_Code"]
            }
        )

        # some values are erroneously above the ceiling value as per scale design
        for k, v in ITEM_MAX.items():
            df[k] = np.clip(df[k], a_min=0, a_max=v)
        df["YMRS_SUM"] = np.sum(
            df[[col for col in df.columns if bool(re.search("YMRS[0-9]", col))]], axis=1
        )
        df["HDRS_SUM"] = np.sum(
            df[[col for col in df.columns if bool(re.search("HDRS[0-9]", col))]], axis=1
        )

        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        df["YMRS_discretized"] = pd.cut(
            df["YMRS_SUM"],
            bins=[
                0,
                7,
                14,
                25,
                60,
            ],  # [0, 7, 14, 25, 60] <- https://clinicaltrials.gov/ct2/show/NCT00931723
            include_lowest=True,
            right=True,
            labels=False,
        )
        # https://pubmed.ncbi.nlm.nih.gov/19624385/
        df["HDRS_discretized"] = pd.cut(
            df["HDRS_SUM"],
            bins=[
                0,
                7,
                14,
                23,
                52,
            ],  # [0, 7, 14, 23, 52] <- https://en.wikipedia.org/wiki/Hamilton_Rating_Scale_for_Depression
            include_lowest=True,
            right=True,
            labels=False,
        )

        ## renaming drop-out patients
        # inc_dict = {4829136: "ME", 5231378: "ME", 4977776: "MDE_MDD", 5420255: "ME", 4734044: "MDE_BD", 5531493: "MX",
        #  5449488: "MDE_BD"} # 4977776 status?
        # for k, v in inc_dict.items():
        #     df.loc[df["NHC"] == k, "status"] = v

        missing_session_ids = []
        for session_id in df["Session_Code"]:
            zip_file = os.path.join(args.data_dir, f"{str(session_id)}.zip")
            # unzip recording to recording folder not found.
            if not os.path.exists(zip_file):
                missing_session_ids.append(session_id)

        missing_session_ids.extend(
            [
                "1390827",
                "1390844",
                "1390845",
                "1390848",
                "1390856",
                "1399678",
                "1380427",
                "1502656",
                "1502666",
                # "1346703",  # T0 and T1 both have same session_id
            ]
        )
        df = df[~df["Session_Code"].isin(missing_session_ids)]
        df.to_excel(
            os.path.join(FILE_DIRECTORY, "TIMEBASE_database_reshaped.xlsx"),
            index=False,
        )
    else:
        df = pd.read_excel(filename)
    return df
