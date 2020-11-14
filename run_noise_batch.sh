#!/bin/bash

# input_spike_mode=$1
# neural_coding=$2
# time_step=$3
# time_step_save_interval=$4
# vth=$5
# tc=$6                 - for TEMPORAL coding
# time_fire_start=$7    - for TEMPORAL coding # integration duration - n x tc
# time_fire_duration=$8 - for TEMPORAL coding # time window - n x tc
# noise_en=$9
# noise_type=$10
# noise_pr=$11

#is_arr=('REAL' 'POISSON' 'WEIGHTED_SPIKE' 'BURST' 'TEMPORAL')
is_arr=('REAL')
#nc_arr=('RATE' 'WEIGHTED_SPIKE' 'BURST' 'TEMPORAL')
#nc_arr=('RATE' 'WEIGHTED_SPIKE' 'BURST')
#nc_arr=('WEIGHTED_SPIKE' 'BURST')
nc_arr=('BURST')
#nc_arr=('TEMPORAL')


# default - dummy
ts_arr=(100)
tssi_arr=(1)
vth_arr=(1.0)
tc_arr=(0)
tfd_n_tc_arr=(0)
tfs_n_div_tfd_arr=(1)

#
tssi=${tssi_arr[0]}


# noise
#noise_en=0     #: false
noise_en=True      #: true
if [ ${noise_en} = True ]
then
    # del
    #noise_type_arr=('DEL')
    #noise_pr_arr=(0.01 0.02 0.05 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

    # jit
    noise_type_arr=('JIT')
    noise_pr_arr=(0.1 0.3 0.5 0.7 0.9 1.0 2.0 4.0 8.0)

else
    noise_type_arr=('NULL')
    noise_pr_arr=(0)
fi



for ((i_is=0;i_is<${#is_arr[@]};i_is++)) do
    is=${is_arr[$i_is]}

    for ((i_nc=0;i_nc<${#nc_arr[@]};i_nc++)) do
        nc=${nc_arr[$i_nc]}

        if [ ${nc} = 'RATE' ]
        then
            ts_arr=(1000)
            #vth_arr=(0.03125 0.0625 0.0125 0.25 0.4 0.5 0.6 0.8 1.0)
            #vth_arr=(0.2 0.4 0.6 0.8 1.0 1.2)
            vth_arr=(0.4 0.6)
        elif [ ${nc} = 'WEIGHTED_SPIKE' ]
        then
            ts_arr=(1500)
            #vth_arr=(0.6 0.8 1.0)
            #vth_arr=(0.2 0.4 0.6 0.8 1.0 1.2)
            #vth_arr=(0.2 0.4 0.6 0.8 1.0 1.2)
            vth_arr=(1.0 1.2)
        elif [ ${nc} = 'BURST' ]
        then
            ts_arr=(1000)
            #vth_arr=(0.03125 0.0625 0.0125 0.25 0.5 1.0)
            #vth_arr=(0.2 0.4 0.6 0.8 1.0 1.2)
            vth_arr=(0.4 0.6)
        elif [ ${nc} = 'TEMPORAL' ]
        then
            ts_arr=(1500)
            #vth_arr=(0.03125 0.0625 0.0125 0.25 0.4 0.5 0.6 0.8 1.0)
            #vth_arr=(0.03125 0.0625 0.125 0.25 0.4 0.5 0.6 0.8 1.0)
            #vth_arr=(0.2 0.4 0.6 0.8 1.0 1.2)
            #vth_arr=(1.2 1.0 0.8 0.6 0.4 0.2)
            vth_arr=(0.8)

            #tc_arr=(1 2 4 8 16)
            tc_arr=(1 2 4)
            #tfd_n_tc_arr=(2 3 4 5)
            tfd_n_tc_arr=(4 5 6)
            #tfs_n_div_tfd_arr=(2 1)
            tfs_n_div_tfd_arr=(2)
        fi


        for ((i_ts=0;i_ts<${#ts_arr[@]};i_ts++)) do
            ts=${ts_arr[$i_ts]}

            for ((i_vth=0;i_vth<${#vth_arr[@]};i_vth++)) do
                vth=${vth_arr[$i_vth]}

                for ((i_tc=0;i_tc<${#tc_arr[@]};i_tc++)) do
                    tc=${tc_arr[$i_tc]}

                    for ((i_tfd_n_tc=0;i_tfd_n_tc<${#tfd_n_tc_arr[@]};i_tfd_n_tc++)) do
                        tfd_n_tc=${tfd_n_tc_arr[$i_tfd_n_tc]}
                        tfd="$((${tc} * ${tfd_n_tc}))"

                        for ((i_tfs_n_tfd=0;i_tfs_n_tfd<${#tfs_n_div_tfd_arr[@]};i_tfs_n_tfd++)) do
                            tfs_n_tfd=${tfs_n_div_tfd_arr[$i_tfs_n_tfd]}
                            tfs="$((${tfd} / ${tfs_n_tfd}))"
                            #tfs=${expr ${tfd}*${tfs_n_tfd} | bc}


                            for ((i_noise_type=0;i_noise_type<${#noise_type_arr[@]};i_noise_type++)) do
                                noise_type=${noise_type_arr[$i_noise_type]}

                                for ((i_noise_pr=0;i_noise_pr<${#noise_pr_arr[@]};i_noise_pr++)) do
                                    noise_pr=${noise_pr_arr[$i_noise_pr]}


                                    echo is: ${is}, nc: ${nc}, ts: ${ts}, tssi: ${tssi}, vth: ${vth}, tc: ${tc}, tfs: ${tfs}, tfd: ${tfd}, noise_en: ${noise_en}, noise_type: ${noise_type}, noise_pr: ${noise_pr}
                                    ./run_noise.sh ${is} ${nc} ${ts} ${tssi} ${vth} ${tc} ${tfs} ${tfd} ${noise_en} ${noise_type} ${noise_pr}
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done