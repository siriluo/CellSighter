import torch
import numpy as np
from torch_geometric.data import Data
import pandas as pd
import cv2

class GraphDataConstructor:
    """
    Construct graph data with precomputed embeddings as node features.
    """
    
    def __init__(self, embedding_model, device='cuda'):
        self.embedding_model = embedding_model
        self.device = device
        self.embedding_model.eval()

        self.coord_path = "/projects/illinois/vetmed/cb/kwang222/mz_jason/crc_coordinate_csv/removed"
    
    def extract_embeddings(self, dataloader, use_encoder=True):
        """
        Extract embeddings from all samples using upstream model.
        
        Args:
            dataloader: DataLoader containing your raw data (images, text, etc.)
            use_encoder: Whether to get the embeddings of only the encoder or encoder + projection head
            
        Returns:
            embeddings: [num_nodes, embedding_dim]
            labels: [num_nodes]
            metadata: Additional information for each node
        """
        embeddings = []
        labels_list = []
        metadata = []
        node_indices = []
        # for metadata, store the image name?
        
        # iterate through dataloader with batch size 1?
        # get coordinates:
        node_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch['image']
                labels = batch['label'] 

                m = batch.get('mask', None)
                if m is not None:
                    images = torch.cat([images, m], dim=1)     

                images = images.cuda(non_blocking=True)

                # Extract embeddings
                if use_encoder:
                    feat = self.embedding_model.encoder(images)
                else:
                    _, _, feat = self.embedding_model(images)

                # mock cell sample coordinates? actual centroid coords vs detected cell coords from crop
                image_name = batch["image_id"][0]
                cell_id = batch["cell_id"][0]
                
                embeddings.append(feat.detach().cpu())
                labels_list.append(labels)
                metadata.append({"cell_id": cell_id, "image_id": image_name})

                node_indices.append(node_idx)
                node_idx += 1
                # when forming the adjacency matrix, use these coordinates for spatial graph construction
        
        embeddings = torch.cat(embeddings, dim=0)
        labels_list = torch.cat(labels_list, dim=0)
        
        return embeddings, labels_list, metadata, node_indices
    
    def construct_graph(self, embeddings, metadata, labels, node_indices, dist=40, edge_construction='knn'):
        """
        Construct graph from embeddings.
        
        Args:
            embeddings: Node feature embeddings [num_nodes, feature_dim]
            labels: Node labels [num_nodes]
            edge_construction: Method to construct edges ('knn', 'radius', 'spatial', 'custom')
            k: Number of nearest neighbors (for knn)
        """
        num_nodes = embeddings.shape[0]


        graph_info_df = pd.DataFrame({
            "CellID": [md["cell_id"] for md in metadata],
            "image_id": [md["image_id"] for md in metadata],
            "label": labels.numpy(),
            "node_index": node_indices,
        })
        print(len(graph_info_df["CellID"]))
        print(len(graph_info_df["image_id"]))
        
        # Construct edges based on similarity/proximity
        edge_index, node_idx_order = self._build_spatial_graph(graph_info_df, dist_threshold=dist) # embeddings, label=labels, metadata=metadata, df=
        
        correct_order_embeddings = embeddings[node_idx_order] # puts embeddings in the correct order based on node indices from graph construction
        correct_order_labels = labels[node_idx_order]


        # Create PyG Data object
        data = Data(
            x=correct_order_embeddings,
            edge_index=edge_index,
            y=correct_order_labels
        )
        
        return data
    
    def _build_spatial_graph(self, df, dist_threshold=40): # embeddings, label, metadata, 
        """Build radius graph based on embedding distance."""

        # use a mask to group things together based on image_id?
        # group metadata by image_id
        # for each group, build the radius graph separately
        # then combine the edge_index together, adjusting the indices accordingly

        image_ids = df["image_id"].unique()

        # calculate adjacency matrix for each image_id group
        cum_cell_num = 0
        edge_indices = []
        node_indices = []
        for img_id in image_ids:
            coords_i = pd.read_csv(f"{self.coord_path}/{img_id}_cell_info.csv")

            img_id_df = df[df['image_id'] == img_id].copy()

            coords_i['CellID'] = coords_i['CellID'].astype(int)
            img_id_df['CellID'] = img_id_df['CellID'].astype(int)

            group_indices = img_id_df['CellID'].values
            group_coords = coords_i.merge(img_id_df, on='CellID')

            x = group_coords['X'].values
            y = group_coords['Y'].values

            cell_coords = np.stack([x, y], axis=1) #.T
            # print("cell coords shape:", cell_coords.shape)

            cell_num = len(group_indices)

            dist_mat = np.zeros([cell_num, cell_num])
            for i in range(cell_num):
                for j in range(i + 1, cell_num):
                    dist = np.linalg.norm((cell_coords[i] - cell_coords[j]), ord=2)
                    if dist < dist_threshold:
                        dist_mat[i, j] = 1

            dist_mat = dist_mat + dist_mat.T + np.identity(cell_num)

            adj_mat = dist_mat

            # use the adj_mat to form edge_index
            edge_index_local = np.array(np.nonzero(adj_mat))

            edge_index_local[0] += cum_cell_num
            edge_index_local[1] += cum_cell_num

            edge_indices.append(edge_index_local)

            # collect all edge_index together
            cum_cell_num = cum_cell_num + cell_num

            node_indices.append(img_id_df["node_index"].values)

        # This mainly just needs the coordinates information. Embeddings will just be collected as node features.

        edge_index = np.concatenate(edge_indices, axis=1)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        node_indices = np.concatenate(node_indices, axis=0)
        
        return edge_index, node_indices
    

    def analyze_dataset(self, dataloader, distance=40):
        """
        Extract embeddings from all samples using upstream model.
        
        Args:
            dataloader: DataLoader containing your raw data (images, text, etc.)
            
        Returns:
            embeddings: [num_nodes, embedding_dim]
            labels: [num_nodes]
            metadata: Additional information for each node
        """
        labels_list = []
        metadata = []
        node_indices = []
        # for metadata, store the image name?
        
        # iterate through dataloader with batch size 1?
        # get coordinates:
        node_idx = 0
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                labels = batch['label'] 

                # mock cell sample coordinates? actual centroid coords vs detected cell coords from crop
                image_name = batch["image_id"][0]
                cell_id = batch["cell_id"][0]
                
                labels_list.append(labels)
                metadata.append({"cell_id": cell_id, "image_id": image_name})

                node_indices.append(node_idx)
                node_idx += 1
                # when forming the adjacency matrix, use these coordinates for spatial graph construction
        
        labels_list = torch.cat(labels_list, dim=0)

        graph_info_df = pd.DataFrame({
            "CellID": [md["cell_id"] for md in metadata],
            "image_id": [md["image_id"] for md in metadata],
            "label": labels_list.numpy(),
            "node_index": node_indices,
        })

        node_num_neighbors, avg_node_neighbors = self.get_graph_statistics(graph_info_df, dist_threshold=distance)
        
        return node_num_neighbors, avg_node_neighbors


    def get_graph_statistics(self, df, dist_threshold=40): # embeddings, label, metadata, 
        """Build radius graph based on embedding distance."""

        # use a mask to group things together based on image_id?
        # group metadata by image_id
        # for each group, build the radius graph separately
        # then combine the edge_index together, adjusting the indices accordingly

        image_ids = df["image_id"].unique()

        # calculate adjacency matrix for each image_id group
        cum_cell_num = 0
        node_num_neighbors = []
        avg_node_neighbors = []
        for img_id in image_ids:
            coords_i = pd.read_csv(f"{self.coord_path}/{img_id}_cell_info.csv")

            img_id_df = df[df['image_id'] == img_id].copy()

            coords_i['CellID'] = coords_i['CellID'].astype(int)
            img_id_df['CellID'] = img_id_df['CellID'].astype(int)

            group_indices = img_id_df['CellID'].values
            group_coords = coords_i.merge(img_id_df, on='CellID')

            x = group_coords['X'].values
            y = group_coords['Y'].values

            cell_coords = np.stack([x, y], axis=1) #.T
            # print("cell coords shape:", cell_coords.shape)

            cell_num = len(group_indices)

            dist_mat = np.zeros([cell_num, cell_num])
            for i in range(cell_num):
                for j in range(i + 1, cell_num):
                    dist = np.linalg.norm((cell_coords[i] - cell_coords[j]), ord=2)
                    if dist < dist_threshold:
                        dist_mat[i, j] = 1

            dist_mat = dist_mat + dist_mat.T + np.identity(cell_num)

            adj_mat = dist_mat

            num_neighbors = np.sum(adj_mat, axis=1)
            avg_num_neighbors = np.mean(num_neighbors)
            node_num_neighbors.append({f"{img_id}_node_neighbors": num_neighbors.tolist()})
            avg_node_neighbors.append({"img_name": f"{img_id}", "avg_num_neighbors": avg_num_neighbors})

        
        return node_num_neighbors, avg_node_neighbors


