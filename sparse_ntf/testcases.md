Sample test cases executed and verified for output. The output of each of the command is obtained by grepping for "relative_error`
### Input file 
- Input should be a sparse tensor file (e.g., `*.tns`) files in text formats
- Refer to FROSTT datasets (http://frostt.io/tensors/)
### Additional CLI Parameters
* `--gpu_offload {0,2}` where {0,2} is for {ALTO,BLCO}-based constrained STF
  * ALTO-based: ALTO Tensor + Constrained update via NVBLAS offload
  * BLCO-based: BLCO Tensor + Constrained update via custom CUDA kernels (end-to-end fully offloaded)
### Supported Constrained Updates
* --algo {0,1,4} -- {MU, HALS, ADMM}
* --algo {2} -- {BPP} -- in developement
### Example command
* `./sparse_ntf -i ~/hpctensor/data/uber.tns -k 64 -t 20 -e 1 -a 0 --gpu_offload 0 | grep relative_error`
=========================
relative_error @it 0=0.942816
relative_error @it 1=0.93279
relative_error @it 2=0.909411
relative_error @it 3=0.885805
relative_error @it 4=0.870971
relative_error @it 5=0.858209
relative_error @it 6=0.848659
relative_error @it 7=0.841467
relative_error @it 8=0.834984
relative_error @it 9=0.82953
relative_error @it 10=0.825095
relative_error @it 11=0.821221
relative_error @it 12=0.817739
relative_error @it 13=0.814683
relative_error @it 14=0.812014
relative_error @it 15=0.80965
relative_error @it 16=0.807538
relative_error @it 17=0.805649
relative_error @it 18=0.803957
relative_error @it 19=0.802417

* `./sparse_ntf -i ~/hpctensor/data/enron.tns -k 16 -t 20 -e 1 -a 4 --gpu_offload 2 | grep relative_error`
=========================
relative_error @it 0=0.90889
relative_error @it 1=0.800748
relative_error @it 2=0.787089
relative_error @it 3=0.782329
relative_error @it 4=0.780818
relative_error @it 5=0.779817
relative_error @it 6=0.779106
relative_error @it 7=0.778603
relative_error @it 8=0.778225
relative_error @it 9=0.777923
relative_error @it 10=0.777698
relative_error @it 11=0.777538
relative_error @it 12=0.777425
relative_error @it 13=0.777346
relative_error @it 14=0.777287
relative_error @it 15=0.777241
relative_error @it 16=0.777206
relative_error @it 17=0.777178
relative_error @it 18=0.777155
relative_error @it 19=0.777137