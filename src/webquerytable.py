import random
import pickle
import datetime
import pandas as pd
from pytorch_transformers import WarmupLinearSchedule

from src.utils.trec_eval import write_trec_result, get_metrics
from src.model.Matching import MatchingModel
from src.utils.dualg_constructor import DualGraph
from src.utils.load_and_process_data import *
from src.utils.reproducibility import set_random_seed

queries = None
tables = None
qtrels = None


def evaluate(config, model, query_id_list, flag):
    qids = []
    docids = []
    gold_rel = []
    pred_rel = []

    model.eval()
    with torch.no_grad():
        for qid in tqdm(query_id_list):
            query = queries["sentence"][qid]
            query_feature = queries["feature"][qid].to("cuda")

            for (tid, rel) in qtrels[qid].items():
                if tid not in tables:
                    continue

                table = tables[tid]
                row_graph = tables[tid]["col_graph"].to("cuda")
                col_graph = tables[tid]["col_graph"].to("cuda")
                node_features = tables[tid]["node_features"].to("cuda")

                score = model(table, query, row_graph, col_graph, node_features, query_feature).item()

                qids.append(qid)
                docids.append(tid)
                gold_rel.append(rel)
                pred_rel.append(score)

    eval_df = pd.DataFrame(data={
        'id_left': qids,
        'id_right': docids,
        'true': gold_rel,
        'pred': pred_rel
    })

    rank_path = './saved/results_webq/trec_rank_{}.txt'.format(flag)
    qrel_path = './saved/results_webq/trec_qrel_{}.txt'.format(flag)
    write_trec_result(eval_df, rank_path, qrel_path)
    metrics = get_metrics('ndcg_cut', rank_path, qrel_path)
    metrics.update(get_metrics('map', rank_path, qrel_path))
    return metrics


def train(config, model, train_query_ids, optimizer, scheduler, loss_func):
    random.shuffle(train_query_ids)
    model.train()

    eloss = 0
    batch_loss = 0
    n_iter = 0
    cnt = 0

    pbar = tqdm(train_query_ids)
    for qid in pbar:
        cnt += 1

        query = queries["sentence"][qid]
        query_feature = queries["feature"][qid].to("cuda")

        logits = []
        label = None
        pos = 0
        for (tid, rel) in qtrels[qid].items():
            if tid not in tables:
                continue

            if rel == 1:
                label = torch.LongTensor([pos]).to("cuda")

            table = tables[tid]
            row_graph = tables[tid]["row_graph"].to("cuda")
            col_graph = tables[tid]["col_graph"].to("cuda")
            node_features = tables[tid]["node_features"].to("cuda")

            logit = model(table, query, row_graph, col_graph, node_features, query_feature)
            logits.append(logit)

            pos += 1

        if label is None or len(logits) < 2:
            print(qid, query)
            continue

        loss = loss_func(torch.cat(logits).view(1, -1), label.view(1))

        batch_loss += loss

        n_iter += 1

        if n_iter % config["batch_size"] == 0 or cnt == len(train_query_ids):
            batch_loss /= config["batch_size"]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            pbar.set_postfix(loss=batch_loss.item())
            optimizer.zero_grad()
            batch_loss = 0

        eloss += loss.item()

    return eloss / len(train_query_ids)


def wqt_train(config):
    set_random_seed()

    global queries, tables, qtrels

    train_queries = load_queries(config["data_dir"], "train_query.txt")
    dev_queries = load_queries(config["data_dir"], "dev_query.txt")
    test_queries = load_queries(config["data_dir"], "test_query.txt")
    tables = load_tables(config["data_dir"])
    qtrels = load_qt_relations(config["data_dir"])

    # remove invalid queries
    train_query_ids = list(train_queries.keys())
    train_query_ids = [x for x in train_query_ids if x in qtrels]
    dev_query_ids = list(dev_queries.keys())
    dev_query_ids = [x for x in dev_query_ids if x in qtrels]
    test_query_ids = list(test_queries.keys())
    test_query_ids = [x for x in test_query_ids if x in qtrels]

    queries = {}
    queries["sentence"] = {}
    queries["sentence"].update(train_queries)
    queries["sentence"].update(dev_queries)
    queries["sentence"].update(test_queries)

    del train_queries, dev_queries, test_queries

    ######## clean the dataset ###########
    invalid_query = []
    invalid_table = []
    missing_pos = []
    for qid in list(queries["sentence"].keys()):
        if qid not in qtrels:
            del queries["sentence"][qid]
            invalid_query.append(qid)
        elif qid not in tables:
            del queries["sentence"][qid]
            invalid_table.append(qid)
        elif qid not in qtrels[qid]:
            qtrels[qid][qid] = 1
            missing_pos.append(qid)
    print("invalid_query", len(invalid_query), invalid_query)
    print("invalid_table", len(invalid_table), invalid_table)
    print("missing_pos", len(missing_pos), missing_pos)

    missing_tables = []
    for qid in list(queries["sentence"].keys()):
        for tid in list(qtrels[qid].keys()):
            if tid not in tables:
                del qtrels[qid][tid]
                missing_tables.append(tid)
    print("missing_tables", len(missing_tables), missing_tables)

    valid_tables = set()
    for qid in qtrels.keys():
        for tid in qtrels[qid].keys():
            valid_tables.add(tid)
    for tid in list(tables.keys()):
        if tid not in valid_tables:
            del tables[tid]

    new_ids = []
    for qid in train_query_ids:
        if qid in queries["sentence"]:
            new_ids.append(qid)
    print("missing train queries", len(train_query_ids) - len(new_ids))
    train_query_ids = new_ids

    new_ids = []
    for qid in dev_query_ids:
        if qid in queries["sentence"]:
            new_ids.append(qid)
    print("missing dev queries", len(dev_query_ids) - len(new_ids))
    dev_query_ids = new_ids

    new_ids = []
    for qid in test_query_ids:
        if qid in queries["sentence"]:
            new_ids.append(qid)
    print("missing test queries", len(test_query_ids) - len(new_ids))
    test_query_ids = new_ids
    #######################################

    constructor = DualGraph(config["fasttext"], "webquerytable")

    queries["feature"] = process_queries(queries["sentence"], constructor)
    process_tables(tables, constructor)
    with open(config['data_dir'] + "table_data.pkl", "wb") as f:
        pickle.dump(tables, f)
    # with open(config['data_dir'] + "table_data.pkl", "rb") as f:
    #     tables = pickle.load(f)

    model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                          bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])


    model = model.to("cuda")

    print(config)
    print(model, flush=True)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam([
        {'params': model.bert.parameters(), 'lr': config["bert_lr"]},
        {'params': (x for x in model.parameters() if x not in set(model.bert.parameters())), 'lr': config["gnn_lr"]}
    ])
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config["warmup_steps"], t_total=config["total_steps"])

    best_metrics = None

    for epoch in range(config['epoch']):
        eloss = train(config, model, train_query_ids, optimizer, scheduler, loss_func)
        dev_metrics = evaluate(config, model, dev_query_ids, "dev_{}".format(epoch))

        if best_metrics is None or dev_metrics[config['key_metric']] > best_metrics[config['key_metric']]:
            best_metrics = dev_metrics
            test_metrics = evaluate(config, model, test_query_ids, "test_{}".format(epoch))
            torch.save(model,"./saved/model_{}.pt".format(epoch))
            print(datetime.datetime.now(), 'epoch', epoch, 'train loss', eloss, 'dev', dev_metrics, 'test',
                  test_metrics, "*", flush=True)
        else:
            print(datetime.datetime.now(), 'epoch', epoch, 'train loss', eloss, 'dev', dev_metrics, flush=True)
