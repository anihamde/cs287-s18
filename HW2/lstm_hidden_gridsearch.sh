n_layers=(1 2 3 4)

h_sizes=(100 200 400 500 600 800 1000 1400 2000)

rm ../../models/HW2/lstm_logs.txt

for i in {0,1,2,3}
do
    for j in {0,1,2,3,4,5,6,7,8}
    do
        echo python3 lstm.py -e 5 -nl ${n_layers[$i]} -hs ${h_sizes[$j]} -m ../../models/HW2/lstm_${i}_${j}.pkl
        python3 lstm.py -e 5 -nl ${n_layers[$i]} -hs ${h_sizes[$j]} -m ../../models/HW2/lstm_${i}_${j}.pkl
    done
done >> ../../models/HW2/lstm_logs.txt
