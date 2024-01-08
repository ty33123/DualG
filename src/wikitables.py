import random
import datetime
import pandas as pd
from sklearn.model_selection import ShuffleSplit
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
        for qid in query_id_list:
            query = queries["sentence"][qid]
            query_feature = queries["feature"][qid].to("cuda")

            for (tid, rel) in qtrels[qid].items():
                table = tables[tid]
                row_graph = tables[tid]["row_graph"].to("cuda")
                col_graph = tables[tid]["col_graph"].to("cuda")
                node_features = tables[tid]["node_features"].to("cuda")

                score = model(table, query, row_graph, col_graph, node_features, query_feature).item()

                qids.append(qid)
                docids.append(tid)
                gold_rel.append(rel)
                pred_rel.append(score * config["relevance_score_scale"])

    eval_df = pd.DataFrame(data={
        'id_left': qids,
        'id_right': docids,
        'true': gold_rel,
        'pred': pred_rel
    })
    rank_path = './saved/results_wikiq/trec_rank_{}.txt'.format(flag)
    qrel_path = './saved/results_wikiq/trec_qrel_{}.txt'.format(flag)
    write_trec_result(eval_df,rank_path, qrel_path)
    metrics = get_metrics('ndcg_cut', rank_path, qrel_path)
    metrics.update(get_metrics('map', rank_path, qrel_path))
    return metrics


def train(config, model, train_pairs, optimizer, scheduler, loss_func, validation_query_ids, n_fold, epoch):
    random.shuffle(train_pairs)
    model.train()

    eloss = 0
    batch_loss = 0
    n_iter = 0
    for (qid, tid, rel) in train_pairs:
        n_iter += 1
        label = rel * 1.0 / config["relevance_score_scale"]

        query = queries["sentence"][qid]
        query_feature = queries["feature"][qid].to("cuda")

        table = tables[tid]
        row_graph = tables[tid]["row_graph"].to("cuda")
        col_graph = tables[tid]["col_graph"].to("cuda")
        node_features = tables[tid]["node_features"].to("cuda")
        prob = model(table, query, row_graph, col_graph, node_features, query_feature)

        loss = loss_func(prob.reshape(-1), torch.FloatTensor([label]).to("cuda"))

        batch_loss += loss

        if n_iter % config["batch_size"] == 0 or n_iter == len(train_pairs):
            batch_loss /= config["batch_size"]
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            scheduler.step()

            optimizer.zero_grad()
            batch_loss = 0
        eloss += loss.item()
    return eloss / len(train_pairs)


def wkt_train(config):
    set_random_seed()
    global queries, tables, qtrels
    # 加载问题、表格以及它们之间的关系打分。
    queries = {}
    queries["sentence"] = load_queries(config["data_dir"])
    tables = load_tables(config["data_dir"])
    qtrels = load_qt_relations(config["data_dir"])

    # 加载模型。
    model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                          bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])
    print(config)
    print(model, flush=True)
    

    constructor = DualGraph(config["fasttext"])

    queries["feature"] = process_queries(queries["sentence"], constructor)
    process_tables(tables, constructor)
    # 处理了tables,加入了dgl_graph、node_features

    loss_func = torch.nn.MSELoss()

    qindex = list(queries["sentence"].keys())
    sample_index = np.array(range(len(qindex))).reshape((-1, 1))

    best_cv_metrics = [None for _ in range(5)]

    seed = 21
    ss = ShuffleSplit(n_splits=5, train_size=0.8, random_state=seed)

    for n_fold, (train_data, validation_data) in enumerate(ss.split(sample_index)):

        train_query_ids = [qindex[idx] for idx in train_data]
        validation_query_ids = [qindex[idx] for idx in validation_data]

        del model
        print("="*280)

        model = MatchingModel(bert_dir=config["bert_dir"], do_lower_case=config["do_lower_case"],
                              bert_size=config["bert_size"], gnn_output_size=config["gnn_size"])

        model = model.to("cuda")

        optimizer = torch.optim.Adam([
            {'params': model.bert.parameters(), 'lr': config["bert_lr"]},
            {'params': (x for x in model.parameters() if x not in set(model.bert.parameters())), 'lr': config["gnn_lr"]}
        ])

        train_pairs = [(qid, tid, rel) for qid in train_query_ids for tid, rel in qtrels[qid].items()]

        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=config['warmup_steps'], t_total=config['total_steps'])
        
        best_metrics = None
        for epoch in range(config['epoch']):
            train(config, model, train_pairs, optimizer, scheduler, loss_func,
                                                validation_query_ids, n_fold, epoch)

            test_metrics = evaluate(config, model, validation_query_ids, "test")

            if best_metrics is None or test_metrics[config['key_metric']] > best_metrics[config['key_metric']]:
                best_metrics = test_metrics
                best_cv_metrics[n_fold] = best_metrics
                print(datetime.datetime.now(), 'epoch', epoch, 'test', test_metrics, "*",
                      flush=True)
            else:
                print(datetime.datetime.now(), 'epoch', epoch, 'test', test_metrics, flush=True)

    avg_metrics = best_cv_metrics[0]
    for key in avg_metrics.keys():
        for metrics in best_cv_metrics[1:]:
            avg_metrics[key] += metrics[key]
        avg_metrics[key] /= 5
    print("5-fold cv scores", avg_metrics)
