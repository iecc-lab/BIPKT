import pandas as pd

from .utils import sta_infos, write_txt, replace_text
KEYS = ["submit_useruid", "kc","problem_pid"]

def read_data_from_csv(read_file, write_file):
    stares = []

    df = pd.read_csv(read_file, low_memory=True, encoding = "utf-8").dropna(subset=["submit_useruid", "kc","problem_pid"])

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    df["index"] = range(df.shape[0])

    df = df.dropna(subset=["submit_useruid", "problem_pid","kc","submit_status","rs","submit_time_cost","problem_difficulty","submit_score","submit_sourceCodeLength","submit_memery_cost"])
    df = df[df['rs'].isin([0,1])]#filter responses
    df['rs'] = df['rs'].astype(int)

    ins, us, qs, cs, avgins, avgcq, na = sta_infos(df, KEYS, stares)
    print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    data = []
    uids = df.submit_useruid.unique()
    problems = df.problem_pid.unique()
    ui_df = df.groupby('submit_useruid', sort=False)


    for ui in ui_df:
        uid, curdf = ui[0], ui[1]
        questions = curdf["problem_pid"].tolist()
        concepts = curdf['kc'].tolist()
        rs = curdf["rs"].astype(int).astype(str).tolist()
        st = curdf['submit_time'].astype(int).astype(str).tolist()
        scl = curdf['submit_sourceCodeLength'].astype(int).astype(str).tolist()
        stc = curdf['submit_time_cost'].astype(int).astype(str).tolist()
        smc = curdf['submit_memery_cost'].astype(int).astype(str).tolist()
        score = curdf['submit_score'].astype(int).astype(str).tolist()
        status = curdf['submit_status'].astype(int).astype(str).tolist()
        problem_difficulty = curdf['problem_difficulty'].astype(int).astype(str).tolist()
        seq_len = len(rs)
        uc = [str(uid), str(seq_len)]
        data.append([uc, questions, concepts, rs, st, scl, stc, smc, score, status, problem_difficulty])
        if len(data) % 1000 == 0:
            print(len(data))
    write_txt(write_file, data)

    print("\n".join(stares))

    return