train-gpu:
	CUDA_VISIBLE_DEVICES=$$(empty-gpu-device) hy ./hae.hy train

log-plot:
	visplot --smoothing 100 -y loss_autoencoder,loss_fake_decoder,loss_variational $$(ls -1 log/*.jsonl | tail -1)
