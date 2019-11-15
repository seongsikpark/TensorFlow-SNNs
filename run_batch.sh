#!/bin/bash

# temporal coding batch experiments

# tc
#tc_arr=(5 10 15 20 25 30)
tc_arr=(15 20)

# time_window - x(tc)
#time_window_arr=(2 3 4)
time_window_arr=(1 2 3 4 5)

# time_fire_start - smaller than time_window

# time_fire_duration -

# time_step - deterministic

# time_step_save_interval
time_step_save_interval=50



for ((i_tc=0;i_tc<${#tc_arr[@]};i_tc++)) do
    tc=${tc_arr[$i_tc]}

    for ((i_tw=0;i_tw<${#time_window_arr[@]};i_tw++)) do
        n_tau=${time_window_arr[$i_tw]}
        tw="$((${tc} * ${n_tau}))"
        #tw=${n_tau}

        #for ((i_tfs=1;i_tfs<=tau;i_tfs++)) do
        #    tfs="$((${tc}*${i_tfs}))"

        tfs=${tw}

        tfd=${tw}

        #ts="$((17 * ${tw}))"
        ts="$((16 * ${tfs} + ${tfd}))"

        tssi=${time_step_save_interval}


        echo tc: ${tc}, tw: ${tw}, tfs: ${tfs}, tfd: ${tfd}, ts: ${ts}, tssi: ${tssi}

        ./run.sh ${tc} ${tw} ${tfs} ${tfd} ${ts} ${tssi}


        #done
    done
done



