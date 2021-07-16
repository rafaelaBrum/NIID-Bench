python partition.py \
	--dataset=generated \
	--batch-size=64 \
	--n_parties=4 \
	--partition=real \
	--beta=0.5\
	--datadir='./data/' \
	--logdir='./logs/' \
	--noise=0\
	--init_seed=0


#python partition.py \
#  --model=mlp \
#	--dataset=generated \
#	--alg=fedavg \
#	--lr=0.01 \
#	--batch-size=64 \
#	--epochs=10 \
#	--n_parties=4 \
#	--rho=0.9 \
#	--comm_round=10 \
#	--partition=real \
#	--device='cpu'\
#	--datadir='./data/' \
#	--logdir='./logs/' \
#	--noise=0\
#	--init_seed=0