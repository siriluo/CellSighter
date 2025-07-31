import sys
sys.path.append(".")
from torch.utils.tensorboard import SummaryWriter
import os
import torch
import argparse
import numpy as np
import pandas as pd
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from gat_model import GAT_model
from mlp_model import GMLP_Model
from model import Model
from data.data import CellCropsDataset
from data.utils import load_crops
from data.transform import train_transform, val_transform
from torch.utils.data import DataLoader, WeightedRandomSampler
import json
from metrics.metrics import Metrics
# from eval import val_epoch


def train_epoch(model, dataloader, optimizer, criterion, epoch, writer, device=None):
    model.train()
    cells = []
    total_loss = 0
    for i, batch in enumerate(dataloader):
        x = batch['image']
        m = batch.get('mask', None)
        if m is not None:
            x = torch.cat([x, m], dim=1)
        x = x.to(device=device)

        y = batch['label'].to(device=device)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        if i % 100 == 0:
            print(f"epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}")
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

    print(f"mean loss: {total_loss / i}")
    return cells


def create_features(model, dataloader, epoch, device=None):
    with torch.no_grad():
        cells_features = []
        model.eval()
        for i, batch in enumerate(dataloader):
            x = batch['image']
            m = batch.get('mask', None)
            if m is not None:
                x = torch.cat([x, m], dim=1)
            x = x.to(device=device)
            # print(x.shape)
            y = batch['label'].to(device=device)
            y_pred = model(x)
            cells_features.append(y_pred)
            if i % 100 == 0:
                print(f"epoch {epoch} | iterate {i} / {len(dataloader)} |")

    return cells_features


def train_graph_epoch(model, dataloader, optimizer, criterion, epoch, writer, device=None):
    model.train()
    cells = []
    total_loss = 0
    total_examples = 0
    for i, batch in enumerate(dataloader):
        y = batch.y[:batch.batch_size].type(torch.LongTensor).to(device=device)

        optimizer.zero_grad()
        y_pred = model(batch.x.to(device), batch.edge_index.to(device), batch_size=batch.batch_size)[:batch.batch_size]

        loss = criterion(y_pred, y)

        loss.backward()
        # for name, param in model.named_parameters():
        #     print(name, param.grad)

        optimizer.step()
        if i % 100 == 0:
            print(f"epoch {epoch} | iterate {i} / {len(dataloader)} | {loss.item()}") #  * batch.batch_size  * batch.batch_size
        writer.add_scalar('Loss/train', loss.item(), epoch * len(dataloader) + i)
        total_loss += loss.item() * batch.batch_size
        total_examples += batch.batch_size

    print(f"mean loss: {total_loss / total_examples}")
    return cells


def val_graph_epoch(model, dataloader, device=None):
    with torch.no_grad():
        model.eval()
        # results = []
        total_loss = 0
        total_examples = 0
        for i, batch in enumerate(dataloader):
            y = batch.y[:batch.batch_size].type(torch.LongTensor).to(device=device)

            y_pred = model(batch.x.to(device), batch.edge_index.to(device), batch_size=batch.batch_size)[:batch.batch_size]
            # results += y_pred.numpy().tolist()

            # Use logits and the label to eval.
            loss = criterion(y_pred, y)
            if i % 500 == 0:
                print(f"Eval {i} / {len(dataloader)}, Loss: {loss.item()}") #  * batch.batch_size  * batch.batch_size  | {loss.item()}
            total_loss += loss.item() * batch.batch_size
            total_examples += batch.batch_size

        print(f"mean loss: {total_loss / total_examples}")
    return # np.array(results) #  cells,


def subsample_const_size(crops, size):
    """
    sample same number of cell from each class
    """
    final_crops = []
    crops = np.array(crops)
    labels = np.array([c._label for c in crops])
    chosen_indices_total = []
    for lbl in np.unique(labels):
        indices = np.argwhere(labels == lbl).flatten()
        if (labels == lbl).sum() < size:
            chosen_indices = indices
        else:
            chosen_indices = np.random.choice(indices, size, replace=False)
        final_crops += crops[chosen_indices].tolist()
        chosen_indices_total += chosen_indices.tolist()
    
    # After sampling the crops, sort the crops based on the image id.
    final_crops = sorted(final_crops, key=lambda crop: (int(crop._image_id[4:6]) - 1, crop._cell_id))

    return final_crops, chosen_indices_total


def define_sampler(crops, hierarchy_match=None):
    """
    Sampler that sample from each cell category equally
    The hierarchy_match defines the cell category for each class.
    if None then each class will be category of it's own.
    """
    labels = np.array([c._label for c in crops])
    if hierarchy_match is not None:
        labels = np.array([hierarchy_match[str(l)] for l in labels])

    unique_labels = np.unique(labels)
    class_sample_count = {t: len(np.where(labels == t)[0]) for t in unique_labels}
    weight = {k: sum(class_sample_count.values()) / v for k, v in class_sample_count.items()}
    samples_weight = np.array([weight[t] for t in labels])
    samples_weight = torch.from_numpy(samples_weight)
    return WeightedRandomSampler(samples_weight.double(), len(samples_weight))


def get_usable_cell_ids(cell_crops=None):
    img_A_list = [[] for _ in range(70)]
    img_B_list = [[] for _ in range(70)]

    if not (cell_crops is None):
        for i, crop in enumerate(cell_crops):
            image_idx = int(crop._image_id[4:6]) - 1
            group_letter = crop._image_id[7:8]
            if group_letter == "A":
                img_A_list[image_idx].append(crop._cell_id - 1)
            elif group_letter == "B":
                img_B_list[image_idx].append(crop._cell_id - 1)

    return img_A_list, img_B_list


# Change the code to instead exclude the specified train_patient_cutoff instead.
def create_edge_indices(label_path_dfs, radius, patient_nums, usable_cell_ids, label_path_group = "A", cell_crops=None): 
    # replace parameters usable cell ids with an outer function? and also add a new parameter for the label path, that way I can get A and B separately

    # Use the cell_id and image_id of the crops to create the edges
    # make sure to use cell_ids - 1
    # While creating edge index list, only include a cell if it is in subsample.
    # create a new usable_cell_ids list from the list of usable cell_crops
    # img_A_list = [[] for _ in range(70)]
    # img_B_list = [[] for _ in range(70)]

    # if not (cell_crops is None):
    #     new_usable_cell_ids = [[] for _ in range(70)]
    #     for i, crop in enumerate(cell_crops):
    #         image_idx = int(crop._image_id[4:6]) - 1
    #         group_letter = crop._image_id[7:8]
    #         if group_letter == "A":
    #             img_A_list[image_idx].append(crop._cell_id - 1)
    #         elif group_letter == "B":
    #             img_B_list[image_idx].append(crop._cell_id - 1)

    #         new_usable_cell_ids[image_idx].append(crop._cell_id - 1)


    # now do again for validation split. Just use last two for validation? Or use validation as an input parameter

    # patient_nums = np.arange(1, 2)
    # For the working cell ids, I will need to make sure they're split between A and B later
    # Or, just make this process accept the stuff or A and B separately?

    patient_cut = train_patient_cutoff+1

    edge_indices_list = []
    num_nodes_list = []
    count = 0
    cum_sum = 0
    for i in patient_nums:
        label_path = f"{label_path_dfs}/patient_{i}/patients_{i}_ct_labels_{label_path_group}.csv"
        # label_path_B = f"{label_path_dfs}/patient_{i}/patients_{i}_ct_labels_B.csv"
        label_paths = [label_path]  # , label_path_B
        # Then, for each image file, create a separate edge index
        # Repeat for A and B
        # if not (cell_crops is None):
        #     usable_cell_ids = new_usable_cell_ids

        for label_path_i in label_paths:
            label_i = pd.read_csv(label_path_i)
            image_names = label_i["File Name"].unique()
            image_names.sort()
            for i, img_name in enumerate(image_names):
                df_new = label_i[label_i['File Name'] == img_name]
                x = df_new["X_withinTile:X_withinTile"].to_numpy()
                y = df_new["Y_withinTile:Y_withinTile"].to_numpy()

                # if not (cell_crops is None):
                #     curr_image_usable_cell_ids = np.array(new_usable_cell_ids[count])
                #     cum_sum += len(curr_image_usable_cell_ids)
                #     x = x[curr_image_usable_cell_ids]
                #     y = y[curr_image_usable_cell_ids]

                #     num_cells = len(x)
                # else:
                curr_image_usable_cell_ids = np.array(usable_cell_ids[count]).astype(int)
                cum_sum += len(curr_image_usable_cell_ids)
                # print(curr_image_usable_cell_ids)
                x = x[curr_image_usable_cell_ids]
                y = y[curr_image_usable_cell_ids]

                num_cells = len(x)
                edge_index = []
                # if one cell and another have an edge, then they will both be within radius of each other, so
                # I don't need to double it each time, only once
                for cell_id in np.arange(num_cells):
                    start_node = np.array([x[cell_id], y[cell_id]])
                    distances = np.sqrt((x - start_node[0]) ** 2 + (y - start_node[1]) ** 2)
                    distances_mask = distances <= radius
                    distances = distances * distances_mask
                    distance_indices = np.where(distances > 0)[0]

                    for idx in distance_indices:
                        edge_index.append([cell_id, idx])

                num_nodes_list.append(num_cells)
                edge_index = np.array(edge_index)
                edge_indices_list.append(edge_index)
                count += 1

    return edge_indices_list, num_nodes_list


# This function fixes the edge indices to follow the required format for the pygeometric graph object
def fix_edge_indices(edge_indices_list, num_nodes_list, starting_num_nodes=0):
    all_edge_indices = torch.from_numpy(edge_indices_list[0]).long()
    cumulative_num_nodes = num_nodes_list[0] + starting_num_nodes
    for i in np.arange(1, len(edge_indices_list)):
        curr_edge_index = edge_indices_list[i]
        curr_edge_index = curr_edge_index + cumulative_num_nodes
        curr_edge_index = torch.from_numpy(curr_edge_index).long()
        all_edge_indices = torch.cat((all_edge_indices, curr_edge_index))
        cumulative_num_nodes = cumulative_num_nodes + num_nodes_list[i]
    print(cumulative_num_nodes)
    all_edge_indices = all_edge_indices.t().contiguous()
    # Maybe first try out to see if the crops are done in order already?
    print(all_edge_indices.shape)

    return all_edge_indices, cumulative_num_nodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Arguments')
    parser.add_argument('--base_path', type=str,
                        help='configuration_path')
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.base_path)
    config_path = os.path.join(args.base_path, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    criterion = torch.nn.CrossEntropyLoss()

    train_crops, val_crops, train_working_cell_ids, val_working_cell_ids = load_crops(config["root_dir"],
                                                                                      config["channels_path"],
                                                                                      config["crop_size"],
                                                                                      config["train_set"],
                                                                                      config["val_set"],
                                                                                      config["to_pad"],
                                                                                      blacklist_channels=config[
                                                                                          "blacklist"])

    # This is all the original code
    train_crops = np.array([c for c in train_crops if c._label >= 0])
    val_crops = np.array([c for c in val_crops if c._label >= 0])
    if "size_data" in config:
        train_crops, chosen_subsample_indices = subsample_const_size(train_crops, config["size_data"])
        if len(set(chosen_subsample_indices)) == len(chosen_subsample_indices):
            print("success")
        else:
            print("duplicate found")
        train_working_cell_ids = np.array(chosen_subsample_indices)
    sampler = define_sampler(train_crops, config["hierarchy_match"])
    shift = 5
    crop_input_size = config["crop_input_size"] if "crop_input_size" in config else 100
    aug = config["aug"] if "aug" in config else True
    training_transform = train_transform(crop_input_size, shift) if aug else val_transform(crop_input_size)

    # get graph edge indices:
    # "size_data": 20000, 
    labels_dfs_path = "/projects/illinois/vetmed/cb/kwang222/processed_cia_crc_ffpe_data/label_csvs"
    train_patient_cutoff = 34 # 34
    patient_nums = np.arange(1, 36)
    patient_nums = np.delete(patient_nums, train_patient_cutoff)

    val_train_split = 2
    patient_nums_train = patient_nums[:-val_train_split]
    patient_nums_val = patient_nums[-val_train_split:]

    cell_distance_cutoff = 60
    # before combing all together, keep each edge index tensor in a separate "list", then when concatenating them together,
    # simply increment the values for each one by the "cumulated number of nodes of all graphs that got collated before the currently processed graph"
    cell_ids_list_A, cell_ids_list_B = get_usable_cell_ids(cell_crops=train_crops)

    edge_indices_list_A, num_nodes_list_A = create_edge_indices(labels_dfs_path, cell_distance_cutoff, patient_nums=patient_nums_train, label_path_group="A",
                                                                usable_cell_ids=cell_ids_list_A, cell_crops=train_crops)
    edge_indices_list_B, num_nodes_list_B = create_edge_indices(labels_dfs_path, cell_distance_cutoff, patient_nums=patient_nums_train, label_path_group="B",
                                                                usable_cell_ids=cell_ids_list_B, cell_crops=train_crops)

    all_edge_indices_A, cum_nodes_A = fix_edge_indices(edge_indices_list_A, num_nodes_list_A)

    all_edge_indices_B, cum_nodes_B = fix_edge_indices(edge_indices_list_B, num_nodes_list_B, starting_num_nodes=cum_nodes_A)

    edge_indices_AB = torch.cat((all_edge_indices_A, all_edge_indices_B), dim=1)

    # Figure out a way to stack them together.

    train_dataset = CellCropsDataset(train_crops, transform=training_transform, mask=True)
    val_dataset = CellCropsDataset(val_crops, transform=val_transform(crop_input_size), mask=True)
    train_dataset_for_eval = CellCropsDataset(train_crops, transform=val_transform(crop_input_size), mask=True)

    device = "cuda"
    num_channels = sum(1 for line in open(config["channels_path"])) + 1 - len(config["blacklist"])
    class_num = config["num_classes"]

    model = Model(num_channels + 1, class_num)

    model = model.to(device=device)

    train_loader = DataLoader(train_dataset, batch_size=64,  # config["batch_size"], maybe use "batch_size" of 4?
                              num_workers=config["num_workers"],
                              sampler=sampler if config["sample_batch"] else None,
                              shuffle=False)  # if config["sample_batch"] else True
    # train_loader_for_eval = DataLoader(train_dataset_for_eval, batch_size=config["batch_size"],
    #                                    num_workers=config["num_workers"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"],
                            num_workers=config["num_workers"], shuffle=False)

    num_features = 1536 # 1536 2560
    # graph_mlp_model = GMLP_Model(in_channels=num_features, hidden_channels=num_features, num_classes=2, num_layers=3, num_heads=8, dropout=0.2)
    # graph_mlp_model = graph_mlp_model.to(device=device)

    gat2_model = GAT_model(in_channels=num_features, hidden_channels=num_features, num_layers=3, num_heads=8, dropout=0.2, num_classes=2)
    gat2_model = gat2_model.to(device=device)
 
    optimizer = torch.optim.Adam(gat2_model.parameters(), lr=config["lr"], weight_decay=1e-4) # 
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    if len(config["val_set"]) > 0:
        # print(val_working_cell_ids)
        edge_indices_list_val, num_nodes_list_val = create_edge_indices(labels_dfs_path, cell_distance_cutoff, patient_nums=patient_nums_val,
                                                                    usable_cell_ids=val_working_cell_ids) # , cell_crops=val_crops
        all_edge_indices_val = fix_edge_indices(edge_indices_list_val, num_nodes_list_val)

        val_labels = np.array([c._label for c in val_crops])
        cell_features_val = create_features(model, val_loader, device=device, epoch=1)
        cell_features_val = torch.cat(cell_features_val, dim=0)

        val_graph_data = Data(x=cell_features_val, edge_index=all_edge_indices_val, y=val_labels)

        kwargs = {'batch_size': 256}
        val_loader_graph = NeighborLoader(val_graph_data, num_neighbors=[16] * 3, shuffle=False, **kwargs) # maybe try 15

    # create the features here
    print(len(train_loader), len(val_loader))
    train_labels = np.array([c._label for c in train_crops])
    print(train_labels.shape)

    cell_features = create_features(model, train_loader, device=device, epoch=1)
    print(len(cell_features))
    cell_features = torch.cat(cell_features, dim=0)
    print(cell_features.shape)

    # Create the labels correctly using train_crops:
    # I can just simply stack all the labels together into one giant 1-d tensor
    total_graph_data = Data(x=cell_features, edge_index=all_edge_indices_A, y=train_labels)

    kwargs = {'batch_size': 256}
    train_loader_graph = NeighborLoader(total_graph_data, num_neighbors=[16] * 3, shuffle=True, **kwargs) # maybe try 15

    # Now that the data loader is ready, I need to create the training loop and get the graph ready.

    for i in range(config["epoch_max"]):
        train_graph_epoch(gat2_model, train_loader_graph, optimizer, criterion, device=device, epoch=i, writer=writer)
        print(f"Epoch {i} done!")
        torch.save(gat2_model.state_dict(), os.path.join(args.base_path, f"./weights_{i}_count.pth"))

        if len(config["val_set"]) > 0:
            # uncomment if you want to eval on validation.
            if (i % 5 == 0) & (i > 0):
                val_graph_epoch(gat2_model, val_loader_graph, device=device)
    #     cells_val, results_val = val_epoch(model, val_loader, device=device)
    #     metrics = Metrics([],
    #                       writer,
    #                       prefix="val")
    #     metrics(cells_val, results_val, i)
    #     metrics.save_results(os.path.join(args.base_path, f"val_results_{i}.csv"), cells_val, results_val)
    #  TODO uncooment to eval on the train as well
    # cells_train, results_train = val_epoch(model, train_loader_for_eval, device=device)
    #  metrics = Metrics(
    #     [],
    #     writer,
    #     prefix="train")
    # metrics(cells_train, results_train, i)
    # metrics.save_results(os.path.join(args.base_path, f"train_results_{i}.csv"), cells_train, results_train)
