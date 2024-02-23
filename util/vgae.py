import torch
from torch_geometric.nn import VGAE, GCNConv

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = GCNConv(in_channels, 2*out_channels)
        self.conv_mu = GCNConv(2*out_channels, out_channels)
        self.conv_logstd = GCNConv(2*out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()

        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

def run_vgae(train_x_tensor, val_x_tensor, test_x_tensor, 
             in_channels, out_channels, device, n_epochs=200):
    model_vgae = VGAE(VariationalGCNEncoder(in_channels, out_channels))
    model_vgae = model_vgae.to(device)
    optimizer = torch.optim.Adam(model_vgae.parameters(), lr=0.01)

    auc_train_list, ap_train_list = [], []
    auc_val_list, ap_val_list = [], []
    auc_test_list, ap_test_list = [], []
    loss_list = []
    
    num_graphs_train = train_x_tensor.shape[0]
    num_graphs_val = val_x_tensor.shape[0]
    num_graphs_test = test_x_tensor.shape[0]
    
    for epoch in range(n_epochs):
        model_vgae.train()
        loss_epoch = 0.0
        auc_train_epoch = 0.0
        ap_train_epoch = 0.0
        for i in range(num_graphs_train):
            optimizer.zero_grad()
            node_feature = train_x_tensor[i, :, :].to(device)
            edge_index = train_adjs_tensor[i].nonzero().t().contiguous().to(device)
            #print(node_feature.shape, edge_index.shape)
            num_nodes_train = node_feature.shape[0]
            
            z = model_vgae.encode(node_feature, edge_index)
            #recon_loss = model.recon_loss(z, data.pos_edge_label_index)
            recon_loss = model_vgae.recon_loss(z, edge_index)
            kl_loss = (1 / num_nodes_train) * model_vgae.kl_loss()
            loss = recon_loss + kl_loss
            loss.backward()
            loss_epoch += loss.item()
            optimizer.step()
    
            neg_edge_label_index = negative_sampling(edge_index, z.size(0))
            auc, ap = model_vgae.test(z, edge_index, neg_edge_label_index)
            auc_train_epoch += auc
            ap_train_epoch += ap
    
        loss_list.append(loss_epoch/num_graphs_train)
    
        auc_train_list.append(auc_train_epoch/num_graphs_train)
        ap_train_list.append(ap_train_epoch/num_graphs_train)
        
        if epoch % 10 == 0:
            #print(f'Epoch: {epoch:03d}, loss: {loss/num_nodes_train:.4f}')
            print(f'Epoch: {epoch:03d}, Loss: {loss_epoch/num_graphs_train:.4f}, AUC: {auc_train_epoch/num_graphs_train:.4f}, AP: {ap_train_epoch/num_graphs_train:.4f}')
        
        model_vgae.eval()
        auc_val_epoch = 0.0
        ap_val_epoch = 0.0
        for i in range(num_graphs_val):
            node_feature = val_x_tensor[i, :, :].to(device)
            edge_index = val_adjs_tensor[i].nonzero().t().contiguous().to(device)
            num_nodes_test = node_feature.shape[0]
            
            z = model_vgae.encode(node_feature, edge_index)
            neg_edge_label_index = negative_sampling(edge_index, z.size(0))
            auc, ap = model_vgae.test(z, edge_index, neg_edge_label_index)
            auc_val_epoch += auc
            ap_val_epoch += ap
        
        auc_val_list.append(auc_val_epoch/num_graphs_val)
        ap_val_list.append(ap_val_epoch/num_graphs_val)
        
        auc_test_epoch = 0.0
        ap_test_epoch = 0.0
        for i in range(num_graphs_test):
            node_feature = test_x_tensor[i, :, :].to(device)
            edge_index = test_adjs_tensor[i].nonzero().t().contiguous().to(device)
            num_nodes_test = node_feature.shape[0]
            
            z = model_vgae.encode(node_feature, edge_index)
            neg_edge_label_index = negative_sampling(edge_index, z.size(0))
            auc, ap = model_vgae.test(z, edge_index, neg_edge_label_index)
            auc_test_epoch += auc
            ap_test_epoch += ap
        
        auc_test_list.append(auc_test_epoch/num_graphs_test)
        ap_test_list.append(ap_test_epoch/num_graphs_test)
        
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:03d}, AUC: {auc_test_epoch/num_graphs_test:.4f}, AP: {ap_test_epoch/num_graphs_test:.4f}')

    return model_vgae