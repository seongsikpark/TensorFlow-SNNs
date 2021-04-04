#!/bin/bash


#epoch=$1
#epoch_start_train_tk=$2
#epoch_start_train_t_int=$3
#epoch_start_train_floor=$4
#epoch_start_train_clip_tw=$5
#epoch_start_loss_enc_spike=$6


#w_loss_enc_spike=$7

#
#bypass_pr=$8
#bypass_target_epoch=$9
#
#cp_mode=$10
#f_training=$11




# training / inference mode
f_training=True
#f_training=False


#epoch=$1
#ep=500
ep=10000
#ep=1

#epoch_start_train_tk=$2
#ep_tk_arr=(0 250 500)
ep_tk_arr=(50)
#ep_tk_arr=(0)

#
#w_tk_arr=(1 0.1 0.01 0.001)
w_tk_arr=(1)

# regularization - temporal kernel decay para
#w_tkr_arr=(0 0.1 0.01 0.001 0.0001)
#w_tkr_arr=(0.00000001)
w_tkr_arr=(0.00001)

# regularization - type
# L2-Z: L2 nrom - zero based
# L2-I: L2 norm - init value based
# L1-I: L2 norm - init value based
t_tkr_arr=(L2-I)

# tk training strategies
# N: (N)one
# R: (R#)ound robin - weight #, rc #, td #
# I: (I)rregular round robin - weight #, tc, td
#s_tk_arr=(N R-1 R-10 I-10)
#s_tk_arr=(R-1)
s_tk_arr=(N)
#s_tk_arr=(I-10)

#
#epoch_start_train_t_int=$3
#ep_enc_int_arr=(0 250 500)
#ep_enc_int_arr=(500)
#ep_enc_int_arr=(0 250)
ep_enc_int_arr=(0)
#ep_enc_int_arr=(250)

#epoch_start_train_floor=$4
#ep_enc_int_fl_arr=(0 250 500)
#ep_enc_int_fl_arr=(0 250)
#ep_enc_int_fl_arr=(500)
ep_enc_int_fl_arr=(10000)
#ep_enc_int_fl_arr=(50)

#epoch_start_train_clip_tw=$5
#ep_dec_prun_arr=(0 250 500)
#ep_dec_prun_arr=(0 250)
ep_dec_prun_arr=(0)
#ep_dec_prun_arr=(200)

#epoch_start_loss_enc_spike=$6
#ep_loss_enc_arr=(0 250 1000)
#ep_loss_enc_arr=(0 250 500)
#ep_loss_enc_arr=(0 250)
ep_loss_enc_arr=(50)
#ep_loss_enc_arr=(10000)
#ep_loss_enc_arr=(0)

# enc_spike - weight
#w_loss_enc_arr=(0.01 0.001)
#w_loss_enc_arr=(1 0.1 0.01 0.001)
#w_loss_enc_arr=(0.1 0.001)
#w_loss_enc_arr=(0.001)
w_loss_enc_arr=(0.0000001)

# enc_spike - n_tw
#nt_loss_enc_arr=(1 2 5 10)
nt_loss_enc_arr=(5)

# enc_spike - kl loss distribution
# beta, gamma, horseshoe
# bn - batchnorm (new)
#dist_loss_enc_arr=(b b2 b3 g h)
#dist_loss_enc_arr=(h)
#dist_loss_enc_arr=(b2)
dist_loss_enc_arr=(bn)

# enc_spike - encoding max spike time mode - f(fixed), nt(n times time window=nt)
#ems_loss_enc_arr=(f n)
ems_loss_enc_arr=(f)

#bypass_pr=$7
#bypass_pr_arr=(0 0.5 1)
#bypass_pr_arr=(0.5 1)
#bypass_pr_arr=(0 -500)
bypass_pr_arr=(0)

#bypass_target_epoch=$8
#bypass_tep_arr=(500)
#bypass_tep_arr=(250 500)
#bypass_tep_arr=(250)
bypass_tep_arr=(50)
#bypass_tep_arr=(200)

# copy model - copy trained model
#cp_model=True
#cp_model=False

# training td
f_td_training=True
#f_td_training=False




for ((i_ep_tk=0;i_ep_tk<${#ep_tk_arr[@]};i_ep_tk++)) do
    ep_tk=${ep_tk_arr[$i_ep_tk]}

    for ((i_w_tk=0;i_w_tk<${#w_tk_arr[@]};i_w_tk++)) do
        w_tk=${w_tk_arr[$i_w_tk]}

        for ((i_w_tkr=0;i_w_tkr<${#w_tkr_arr[@]};i_w_tkr++)) do
            w_tkr=${w_tkr_arr[$i_w_tkr]}

            for ((i_t_tkr=0;i_t_tkr<${#t_tkr_arr[@]};i_t_tkr++)) do
                t_tkr=${t_tkr_arr[$i_t_tkr]}


                for ((i_tks=0;i_tks<${#s_tk_arr[@]};i_tks++)) do
                    tks=${s_tk_arr[$i_tks]}

                    for ((i_ep_enc_int=0;i_ep_enc_int<${#ep_enc_int_arr[@]};i_ep_enc_int++)) do
                        ep_enc_int=${ep_enc_int_arr[$i_ep_enc_int]}

                        for ((i_ep_enc_int_fl=0;i_ep_enc_int_fl<${#ep_enc_int_fl_arr[@]};i_ep_enc_int_fl++)) do
                            ep_enc_int_fl=${ep_enc_int_fl_arr[$i_ep_enc_int_fl]}

                            for ((i_ep_dec_prun=0;i_ep_dec_prun<${#ep_dec_prun_arr[@]};i_ep_dec_prun++)) do
                                ep_dec_prun=${ep_dec_prun_arr[$i_ep_dec_prun]}

                                for ((i_ep_loss_enc=0;i_ep_loss_enc<${#ep_loss_enc_arr[@]};i_ep_loss_enc++)) do
                                    ep_loss_enc=${ep_loss_enc_arr[$i_ep_loss_enc]}

                                    for ((i_w_loss_enc=0;i_w_loss_enc<${#w_loss_enc_arr[@]};i_w_loss_enc++)) do
                                        w_loss_enc=${w_loss_enc_arr[$i_w_loss_enc]}

                                        for ((i_nt_loss_enc=0;i_nt_loss_enc<${#nt_loss_enc_arr[@]};i_nt_loss_enc++)) do
                                            nt_loss_enc=${nt_loss_enc_arr[$i_nt_loss_enc]}

                                            for ((i_dist_loss_enc=0;i_dist_loss_enc<${#dist_loss_enc_arr[@]};i_dist_loss_enc++)) do
                                                dist_loss_enc=${dist_loss_enc_arr[$i_dist_loss_enc]}

                                                for ((i_ems_loss_enc=0;i_ems_loss_enc<${#ems_loss_enc_arr[@]};i_ems_loss_enc++)) do
                                                    ems_loss_enc=${ems_loss_enc_arr[$i_ems_loss_enc]}

                                                    for ((i_bypass_pr=0;i_bypass_pr<${#bypass_pr_arr[@]};i_bypass_pr++)) do
                                                        bypass_pr=${bypass_pr_arr[$i_bypass_pr]}

                                                        for ((i_bypass_tep=0;i_bypass_tep<${#bypass_tep_arr[@]};i_bypass_tep++)) do
                                                            bypass_tep=${bypass_tep_arr[$i_bypass_tep]}

                                                             echo training_mode: ${f_training}
                                                             echo ep: ${ep}, tk: ${ep_tk}, w_tk: ${w_tk}, w_tkr: ${w_tkr}, t_tkr: ${t_tkr}, tks: ${tks}, int: ${ep_enc_int}, ce: ${ep_enc_int_fl}, cl: ${ep_dec_prun}, le: ${ep_loss_enc}, lew: ${w_loss_enc}, nt: ${nt_loss_enc}, led: ${dist_loss_enc}, ems: ${ems_loss_enc}, bp: ${bypass_pr}, bt: ${bypass_tep}

                                                            #./run_enc_dec_tk.sh ${ep} ${ep_tk} ${ep_enc_int} ${ep_enc_int_fl} ${ep_dec_prun} ${ep_loss_enc} ${bypass_pr} ${bypass_tep} ${cp_model} ${f_training}
                                                            ./run_enc_dec_tk.sh ${ep} ${ep_tk} ${w_tk} ${w_tkr} ${t_tkr} ${tks} ${ep_enc_int} ${ep_enc_int_fl} ${ep_dec_prun} ${ep_loss_enc} ${w_loss_enc} ${nt_loss_enc} ${dist_loss_enc} ${ems_loss_enc} ${bypass_pr} ${bypass_tep} ${f_training} ${f_td_training}
                                                        done
                                                    done
                                                done
                                            done
                                        done
                                    done
                                done
                            done
                        done
                    done
                done
            done
        done
    done
done



