python train.py --dataset 'Cora' --K 70 --pri_hop 2 --sup_hop 1 --global_hop 10 \
--lr_p 0.001 --num_hidden 256 --tau 0.8 --eps1 1.0 --eps2 1.0 --drop_edge_rate 0.1 --drop_kg_edge_rate 0.2
python train.py --dataset 'CiteSeer' --K 70 --pri_hop 2 --sup_hop 1 --global_hop 5 \
--lr_p 0.001 --num_hidden 256 --tau 1.6 --eps1 1.0 --eps2 1.0 --drop_edge_rate 0.3 --drop_kg_edge_rate 0.7
python train.py --dataset 'PubMed' --K 70 --pri_hop 2 --sup_hop 1 --global_hop 30 \
--lr_p 0.01 --num_hidden 256 --tau 1.0 --eps1 1.0 --eps2 1.0 --drop_edge_rate 0.1 --drop_kg_edge_rate 0.5
python train.py --dataset 'Cornell' --K 5 --pri_hop 2 --sup_hop 1 --global_hop 3 \
--lr_p 0.001 --num_hidden 128 --tau 6.0 --eps1 0.0 --eps2 0.0 --drop_edge_rate 0.1 --drop_kg_edge_rate 0.7
python train.py --dataset 'Texas' --K 50 --pri_hop 2 --sup_hop 1 --global_hop 5 \
--lr_p 0.001 --num_hidden 128 --tau 0.2 --eps1 1.0 --eps2 1.0 --drop_edge_rate 0.1 --drop_kg_edge_rate 0.4
python train.py --dataset 'Wisconsin' --K 50 --pri_hop 2 --sup_hop 1 --global_hop 3 \
--lr_p 0.001 --num_hidden 128 --tau 0.5 --eps1 0.0 --eps2 0.0 --drop_edge_rate 0.1 --drop_kg_edge_rate 0.4
python train.py --dataset 'Actor' --K 5 --pri_hop 1 --sup_hop 1 --global_hop 1 \
--lr_p 0.0001 --num_hidden 256 --tau 6.0 --eps1 0.0 --eps2 0.0 --drop_edge_rate 0.2 --drop_kg_edge_rate 0.6