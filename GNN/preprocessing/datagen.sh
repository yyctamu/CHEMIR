N=10000
run=true # false
for ((i = 0; i <= 25; i++)); do
    start=$((i * N))
    stop=$(((i + 1) * N))
    echo "Molecules ${start}-$((stop - 1))"
    cmd="echo Molecules "${start}"-"$((stop - 1))"; "
    cmd+="conda activate ir; python preprocessed_vocab.py --start=${start} --stop=${stop}"
    echo $cmd
    if $run; then
        echo RUNNING!
        screen_name=gen${start}_${stop}
        screen -dmS ${screen_name}
        screen -S ${screen_name} -X stuff "${cmd}; echo DONE.$(printf \\r)"
    fi
done