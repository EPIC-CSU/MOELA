for app in {bfs, bp, gau, hotspot, kmeans, nw, pf, sc, srad}
do
    for random_seed in {1,2,5,10,20}
    do
        echo "python MOELA.py $app $random_seed"
        python MOELA.py $app $random_seed
    done
done