# _*_ coding:utf-8 _*_
"""
 @author: LaiJinHan
 @time：2019/8/7 23:21
"""
import numpy as np


def load_parameters():
    w1 = np.array([[[[2.65114754e-01, 1.71473354e-01, 5.85026145e-01,
                       3.79897416e-01, -6.38470471e-01, 3.53276759e-01,
                       2.28844926e-01, 5.98939240e-01],
                      [6.20798647e-01, 4.72428888e-01, 2.04147249e-01,
                       9.64657784e-01, -2.92289704e-01, -7.05243573e-02,
                       -3.23324084e-01, 8.22710171e-02],
                      [1.09127653e+00, 4.33395654e-01, 4.25810993e-01,
                       1.29072165e+00, -7.03032255e-01, -4.85596746e-01,
                       -5.99819839e-01, -3.46896678e-01]],

                     [[-6.12088889e-02, -2.01985374e-01, 3.96462828e-01,
                       5.78983963e-01, -2.82581121e-01, 1.19130865e-01,
                       1.19928099e-01, 2.54651636e-01],
                      [5.65950453e-01, -1.25761986e-01, 5.28835952e-01,
                       9.57980454e-01, -3.87472570e-01, -3.46199214e-01,
                       -9.95197594e-02, -1.54194413e-02],
                      [8.62052381e-01, -2.22849503e-01, 4.91660058e-01,
                       1.32679582e+00, -2.31864005e-01, -5.61172068e-01,
                       -3.88903052e-01, -6.22827947e-01]],

                     [[3.88902456e-01, -5.41672528e-01, 2.96558440e-01,
                       -1.58592276e-02, -6.47086725e-02, 1.03312872e-01,
                       1.57973856e-01, -1.09994836e-01],
                      [3.86807352e-01, -7.85594702e-01, 2.59546071e-01,
                       3.80616784e-02, -4.40064996e-01, -1.70494273e-01,
                       3.47007036e-01, -6.03560269e-01],
                      [1.02404964e+00, -1.01177680e+00, 2.75547177e-01,
                       -1.24945529e-01, -6.21036947e-01, -3.35497409e-01,
                       3.38591605e-01, -8.58285487e-01]],

                     [[4.06205177e-01, -3.84070128e-02, 2.52334237e-01,
                       -1.43057257e-01, 1.58749949e-02, 4.70315307e-01,
                       4.17050570e-01, 9.74323601e-02],
                      [6.07388496e-01, -9.23850596e-01, -4.47375596e-01,
                       -7.62789905e-01, -3.27618062e-01, 5.01315534e-01,
                       3.32703173e-01, -2.32586861e-01],
                      [9.01641488e-01, -1.26528990e+00, -6.72339916e-01,
                       -5.84576726e-01, -4.54487443e-01, 2.67813206e-01,
                       6.60709560e-01, -2.85931081e-01]]],

                    [[[2.34028436e-02, 3.07073355e-01, 3.79475206e-01,
                       3.68775487e-01, -5.17509580e-01, 2.94221789e-01,
                       3.63901168e-01, 5.55191576e-01],
                      [5.57209134e-01, 5.52807868e-01, 1.42017782e-01,
                       7.76009619e-01, -7.51562834e-01, -1.95948988e-01,
                       -2.70642638e-01, 3.11052483e-02],
                      [8.34234595e-01, 7.02082396e-01, -1.30136376e-02,
                       1.00349581e+00, -6.57386065e-01, -2.99307883e-01,
                       -5.91448784e-01, -3.14140052e-01]],

                     [[3.06290418e-01, -4.94170189e-02, 2.41752416e-01,
                       4.10424352e-01, 5.67888379e-01, 1.85816899e-01,
                       -4.82525229e-02, 5.49122989e-01],
                      [4.57351774e-01, 7.01577589e-02, -2.72085309e-01,
                       6.64298058e-01, 6.17597580e-01, -3.34590226e-01,
                       -5.32154202e-01, 9.62016210e-02],
                      [9.80453312e-01, -6.69233352e-02, -6.01529241e-01,
                       9.93960142e-01, 3.72841328e-01, -6.80166543e-01,
                       -6.02077842e-01, -9.47109535e-02]],

                     [[3.68882567e-01, -4.59289819e-01, -3.72673385e-02,
                       -2.46379316e-01, 4.10051286e-01, 2.35478938e-01,
                       -7.93218985e-03, 2.91273206e-01],
                      [4.47894007e-01, -8.04806948e-01, -4.36652720e-01,
                       -5.33168018e-01, 3.59417796e-01, 2.59531531e-02,
                       7.82906264e-02, -8.91387835e-02],
                      [7.42124498e-01, -8.88203621e-01, -6.64018989e-01,
                       -5.68923831e-01, 4.20177877e-01, -2.77536839e-01,
                       1.61869660e-01, -4.36954081e-01]],

                     [[3.26999217e-01, -6.36859298e-01, 4.13952544e-02,
                       -4.37484384e-01, 3.94652963e-01, 4.52040404e-01,
                       3.31676155e-01, 5.55869699e-01],
                      [6.78145766e-01, -1.06411994e+00, -6.61713958e-01,
                       -9.76918340e-01, 3.26968282e-01, 4.12274241e-01,
                       3.44010770e-01, 4.15236592e-01],
                      [9.39890623e-01, -1.15872729e+00, -7.67525434e-01,
                       -1.15665209e+00, 4.60388027e-02, 4.58378255e-01,
                       6.86515570e-01, 9.98292193e-02]]],

                    [[[-5.20857573e-02, 3.71165782e-01, 1.64926201e-01,
                       3.29749554e-01, -7.03664422e-01, 3.18443537e-01,
                       4.08304721e-01, 3.87978673e-01],
                      [3.42190921e-01, 6.04106665e-01, -5.36877632e-01,
                       7.33962655e-01, -6.47901058e-01, -2.14490190e-01,
                       -2.70439506e-01, 1.32940993e-01],
                      [6.24150395e-01, 8.30262959e-01, -6.90799713e-01,
                       1.09286582e+00, -6.96399093e-01, -3.00571471e-01,
                       -5.62797964e-01, -5.22568345e-01]],

                     [[3.67487103e-01, 2.25860745e-01, 1.23252906e-01,
                       -3.52630839e-02, 5.30553102e-01, 1.26785979e-01,
                       -1.06790453e-01, 1.68714166e-01],
                      [4.06276017e-01, 5.35242498e-01, -4.54861164e-01,
                       3.05914551e-01, 3.39906782e-01, -3.82861167e-01,
                       -6.44552708e-01, 2.71540247e-02],
                      [7.02725470e-01, 6.27065599e-01, -4.47445005e-01,
                       5.12073100e-01, 2.80579448e-01, -4.09905344e-01,
                       -5.76366484e-01, -2.87884831e-01]],

                     [[1.26140252e-01, 2.23195106e-01, 4.33462471e-01,
                       1.17283231e-02, 5.62580526e-01, 7.30021968e-02,
                       1.80909649e-01, 2.28596151e-01],
                      [4.51059759e-01, -2.05763825e-03, -1.92636445e-01,
                       -4.77184057e-01, 4.52519774e-01, 6.68468252e-02,
                       4.84847743e-03, -9.54811499e-02],
                      [7.97916710e-01, 1.33409247e-01, -4.36267763e-01,
                       -7.21878767e-01, 1.42288685e-01, -9.46366340e-02,
                       -6.00226820e-02, -4.43223119e-01]],

                     [[-3.00592668e-02, -3.57061923e-01, 6.04770184e-01,
                       -9.78420079e-02, 1.75183058e-01, 4.84499782e-01,
                       3.72625440e-01, 5.72279930e-01],
                      [3.92926097e-01, -3.21173787e-01, 3.17040116e-01,
                       -6.44956708e-01, 4.46259022e-01, 4.46104199e-01,
                       5.36398351e-01, 5.89773655e-01],
                      [5.47751188e-01, -4.47539449e-01, 2.46318460e-01,
                       -1.15090585e+00, 3.02258670e-01, 8.01270478e-04,
                       6.38498545e-01, 3.91901374e-01]]],

                    [[[2.32205644e-01, 4.87887233e-01, 2.79140711e-01,
                       3.17689240e-01, -8.54038954e-01, 3.44066679e-01,
                       3.31940770e-01, 5.44930175e-02],
                      [6.49126589e-01, 4.61377144e-01, -4.85612452e-01,
                       5.32174230e-01, -8.65815401e-01, -9.15656015e-02,
                       -2.32011423e-01, -7.91590437e-02],
                      [7.28047132e-01, 8.81137609e-01, -5.49918413e-01,
                       5.23943245e-01, -1.06543291e+00, -4.40200478e-01,
                       -5.16724229e-01, -2.26846933e-01]],

                     [[1.42950863e-01, 4.77007836e-01, 2.96925575e-01,
                       2.88927644e-01, 1.34868190e-01, 4.30842005e-02,
                       1.18657961e-01, 2.69699872e-01],
                      [7.71451592e-01, 4.39420938e-01, -1.82391144e-02,
                       1.15500100e-01, -2.92119741e-01, -2.98858732e-01,
                       -5.10247529e-01, -2.67494261e-01],
                      [9.66260433e-01, 6.02574825e-01, -2.95238137e-01,
                       1.04644962e-01, -2.65622646e-01, -2.94850320e-01,
                       -5.68242431e-01, -4.38161761e-01]],

                     [[3.84501338e-01, -1.09953053e-01, 4.41497624e-01,
                       4.88956459e-02, 3.51353079e-01, 3.24565560e-01,
                       2.82337338e-01, -5.41692436e-01],
                      [6.00168884e-01, 9.05448478e-03, 1.74147367e-01,
                       -4.04293478e-01, 5.54343104e-01, -3.76262367e-02,
                       -7.39296079e-02, -7.63410091e-01],
                      [8.32856417e-01, 5.21085382e-01, -6.61065578e-02,
                       -5.85432470e-01, 1.57932088e-01, -1.91697925e-01,
                       -3.12367797e-01, -1.11956501e+00]],

                     [[1.48125693e-01, -2.15425104e-01, 5.03809452e-01,
                       6.87214583e-02, 6.28045142e-01, 2.83450633e-01,
                       4.81729418e-01, -1.40209422e-01],
                      [5.83751559e-01, -2.38338321e-01, 7.09030256e-02,
                       -5.86677074e-01, 6.19820058e-01, 1.91431463e-01,
                       3.42545837e-01, -2.31310338e-01],
                      [7.46512949e-01, -5.70594221e-02, 4.03611250e-02,
                       -1.10769200e+00, 2.58811444e-01, 1.88056663e-01,
                       2.81772852e-01, -4.31770414e-01]]]])
    w2 = np.array([[[[1.56113073e-01, -1.98769003e-01, 4.70830232e-01,
                       -3.57641995e-01, -4.15169001e-01, 6.49090856e-02,
                       2.27750331e-01, 5.87958753e-01, 4.91985917e-01,
                       2.19631359e-01, -3.59677881e-01, 2.46099204e-01,
                       4.90624420e-02, 3.96059513e-01, 7.78452307e-02,
                       -2.18071699e-01],
                      [7.32831717e-01, 8.60072494e-01, -1.29922259e+00,
                       -5.74042022e-01, -7.72130415e-02, 8.11291099e-01,
                       -7.03326881e-01, -3.94776523e-01, 3.29526484e-01,
                       2.42443994e-01, -3.03993881e-01, 7.90083632e-02,
                       3.09129641e-03, -5.47737360e-01, -4.17315930e-01,
                       -7.14715347e-02],
                      [5.22701800e-01, 8.38704705e-01, 3.84854853e-01,
                       1.17415643e+00, 2.38014206e-01, -5.38191080e-01,
                       5.85286021e-01, -1.75569057e+00, 6.79138452e-02,
                       -4.43329722e-01, -5.48510253e-01, -2.97255993e-01,
                       4.48307008e-01, 8.61598611e-01, 7.00017631e-01,
                       -4.27215636e-01],
                      [8.97251487e-01, 2.16941401e-01, -5.06638646e-01,
                       -9.30712283e-01, 2.09662303e-01, -1.04208574e-01,
                       1.84326828e-01, -8.18025395e-02, 1.63553968e-01,
                       2.94104189e-01, -1.34918420e-02, -2.40335301e-01,
                       -1.06178053e-01, 1.12648094e+00, 3.52049351e-01,
                       2.18986481e-01],
                      [-1.18811987e-01, 6.73807561e-01, 1.61706254e-01,
                       -2.51826495e-01, -1.85887909e+00, -2.71543741e-01,
                       -1.05679765e-01, -6.70956612e-01, 1.12610018e+00,
                       -1.39230037e+00, 2.69315600e-01, 3.33366603e-01,
                       -1.75409704e-01, 4.39211875e-02, 3.66165847e-01,
                       4.49901789e-01],
                      [3.25761706e-01, -4.02296275e-01, 2.32177421e-01,
                       5.13448298e-01, -3.13954800e-01, 6.49242282e-01,
                       -1.33404481e+00, -3.10294360e-01, 4.72167820e-01,
                       -2.15054035e+00, -1.21897139e-01, -1.44621208e-01,
                       4.64150220e-01, 6.65002167e-01, 1.60852218e+00,
                       5.52410960e-01],
                      [3.18102777e-01, 1.13014512e-01, 8.97452354e-01,
                       7.04243481e-01, 1.73673987e-01, 2.28874311e-01,
                       -7.40680754e-01, 5.12784362e-01, 6.61380947e-01,
                       -6.73337877e-01, 1.69056922e-01, -5.07666588e-01,
                       8.36169645e-02, 1.08898234e+00, 1.16001534e+00,
                       2.13893980e-01],
                      [-5.28608859e-01, 1.68490732e+00, -1.44704580e+00,
                       3.99007738e-01, -2.89978892e-01, 1.77426040e+00,
                       -1.50662589e+00, -6.09810889e-01, 2.05695048e-01,
                       -9.10497069e-01, 1.10813928e+00, -1.09607542e+00,
                       -2.23897696e+00, -2.42483184e-01, 1.78227460e+00,
                       -4.47318435e-01]],

                     [[-3.71094376e-01, 3.20944518e-01, -2.14199107e-02,
                       3.64276439e-01, 2.25007325e-01, -8.99327081e-03,
                       2.41367891e-01, 2.72219360e-01, 5.53533256e-01,
                       -2.58383714e-02, -1.97426483e-01, 1.04771115e-01,
                       -1.38591155e-01, 2.43189365e-01, 1.60236478e-01,
                       2.69600123e-01],
                      [3.03542852e-01, -6.02318168e-01, -3.74248654e-01,
                       -9.85280633e-01, 5.83503008e-01, 2.93818146e-01,
                       -6.82772219e-01, -7.12345839e-02, 4.97405529e-01,
                       -3.28499585e-01, -3.38238686e-01, -5.04521310e-01,
                       -7.14358211e-01, -4.39528704e-01, 2.61051238e-01,
                       -1.49815321e-01],
                      [7.85028040e-01, 3.04626953e-02, 2.81655520e-01,
                       1.11710250e+00, 1.12998903e+00, -1.19309032e+00,
                       1.53837705e+00, -9.41934705e-01, 1.98342547e-01,
                       -3.06816429e-01, -2.52186120e-01, -5.74888766e-01,
                       -6.56273007e-01, 1.86415136e+00, -3.46002489e-01,
                       7.83719778e-01],
                      [4.48563010e-01, 5.09316958e-02, -1.56589210e-01,
                       1.26209497e-01, 4.02844697e-01, 2.15432450e-01,
                       1.18960693e-01, -2.33580098e-01, 4.89008456e-01,
                       -1.06388342e+00, -1.68563008e-01, 8.59719962e-02,
                       -1.03511536e+00, 5.98409295e-01, 3.44921082e-01,
                       -1.33083686e-02],
                      [1.52373806e-01, -3.48460257e-01, -1.17207718e+00,
                       -3.86067659e-01, -2.13226713e-02, -4.15320009e-01,
                       -5.44290960e-01, -1.33329046e+00, 5.45920730e-01,
                       1.92373419e+00, -9.34103608e-01, 8.65575671e-01,
                       2.42305025e-01, -7.07234442e-02, 1.80830479e-01,
                       -4.26090032e-01],
                      [-6.77992636e-03, 1.12054312e+00, -1.50164545e+00,
                       2.99066007e-01, -5.83538562e-02, 6.93973184e-01,
                       -1.06059408e+00, -1.62734938e+00, 3.61241370e-01,
                       5.25393724e-01, -7.15857625e-01, -2.27659312e-03,
                       7.32221603e-01, 9.73258138e-01, -1.20876320e-01,
                       7.27227449e-01],
                      [-2.22448274e-01, 1.35488069e+00, -7.80423522e-01,
                       2.70400226e-01, 2.85224140e-01, 4.82153654e-01,
                       -9.20184374e-01, -4.73030150e-01, 3.46626550e-01,
                       7.33456314e-01, 9.57560465e-02, -2.90256828e-01,
                       1.19360767e-01, 1.17564201e+00, 4.79488298e-02,
                       1.03508174e+00],
                      [7.10524261e-01, 5.56355953e-01, 7.30948627e-01,
                       -7.01271044e-03, -1.51285201e-01, 6.27313614e-01,
                       9.72711563e-01, -1.06307232e+00, -1.18118703e+00,
                       -1.56023487e-01, 5.53925782e-02, -2.08826518e+00,
                       -8.35893393e-01, -2.81497508e-01, 3.30977231e-01,
                       1.73399806e+00]]],

                    [[[-2.87029389e-02, 1.93254024e-01, 3.67695868e-01,
                       -3.49062741e-01, -1.55377239e-01, 1.96651623e-01,
                       6.24115430e-02, -1.03679664e-01, -1.59319222e-01,
                       -3.95336933e-02, 1.80474922e-01, 2.24800348e-01,
                       -1.67238757e-01, 7.49830723e-01, 2.64426142e-01,
                       -1.28947929e-01],
                      [1.45937026e-01, -1.45374507e-01, 5.15988410e-01,
                       -4.68714759e-02, -1.46923512e-01, -1.36159077e-01,
                       -4.19243902e-01, -5.19453958e-02, -2.06357062e-01,
                       3.11676450e-02, 4.23891455e-01, -1.51272222e-01,
                       3.97620261e-01, -2.67906040e-01, 1.13541448e+00,
                       -5.69073617e-01],
                      [5.07134557e-01, -1.68390512e+00, 1.07309496e+00,
                       1.73640579e-01, 3.56938094e-01, -9.99873400e-01,
                       9.72240984e-01, 3.23066413e-01, -2.21808657e-01,
                       -3.02471232e+00, -3.96382123e-01, -8.16802025e-01,
                       5.15240058e-02, 1.31601647e-01, 2.61719096e-02,
                       3.72212172e-01],
                      [-1.12172866e+00, 5.11245169e-02, 5.91079235e-01,
                       1.63895607e-01, -3.31773907e-01, -4.20971155e-01,
                       3.04672033e-01, 2.15606555e-01, 1.69729665e-01,
                       3.35167825e-01, -3.20519805e-02, 8.42255056e-01,
                       5.24530351e-01, 1.65035680e-01, 2.22388357e-01,
                       6.17196441e-01],
                      [-6.75087571e-01, -4.99137402e-01, -1.58450767e-01,
                       7.83149421e-01, -2.58645248e+00, -8.18478882e-01,
                       -1.31679118e-01, -4.55555469e-01, -1.18514761e-01,
                       2.03569269e+00, 5.89005947e-01, -1.36021137e-01,
                       -6.50370896e-01, 6.22414291e-01, -3.76772434e-01,
                       -4.26660448e-01],
                      [4.87681508e-01, -1.62130737e+00, 1.48678446e+00,
                       -1.53821811e-01, 2.89235413e-01, -1.69190300e+00,
                       2.40424395e-01, -1.04223514e+00, 7.95997530e-02,
                       -2.21339560e+00, 3.31866801e-01, -2.25221664e-01,
                       1.01046324e+00, 5.70529699e-01, 2.69272000e-01,
                       4.10127342e-01],
                      [7.90598333e-01, -9.57797766e-01, 1.60157287e+00,
                       -1.46122068e-01, 9.06121284e-02, -1.45970070e+00,
                       -4.55247223e-01, -8.76896977e-01, -1.04088292e-01,
                       -5.28640270e-01, -1.77459002e-01, 1.97813481e-01,
                       1.23907745e+00, 1.00129437e+00, 2.84468293e-01,
                       3.25912952e-01],
                      [-2.46719646e+00, -2.96640038e+00, -4.41233486e-01,
                       2.94985712e-01, -1.86277056e+00, -2.02528286e+00,
                       -1.19304645e+00, -1.46545243e+00, -1.82812488e+00,
                       -4.66369057e+00, 5.20379841e-01, -1.00201213e+00,
                       -1.48405623e+00, -4.84592408e-01, 1.05849016e+00,
                       -1.57549717e-02]],

                     [[-1.61164850e-01, -1.42461777e-01, 2.18365401e-01,
                       -3.73686075e-01, -7.25885388e-03, -8.41623768e-02,
                       -1.78324148e-01, -3.81332487e-02, -8.54831859e-02,
                       -4.99224991e-01, 2.36034006e-01, 5.49321398e-02,
                       3.19878131e-01, 6.01155043e-01, 3.75984550e-01,
                       -2.08983809e-01],
                      [1.30668744e-01, 5.48316181e-01, -4.17289972e-01,
                       1.17655387e-02, -6.36877418e-01, 3.85815769e-01,
                       5.08268952e-01, -2.43818140e+00, 1.17817819e-01,
                       -3.89645994e-01, -1.71364427e-01, -2.44834162e-02,
                       -9.63756621e-01, -4.40041214e-01, -1.59307683e+00,
                       -2.81169832e-01],
                      [1.30244541e+00, 2.57179499e-01, 2.35649887e-02,
                       2.29756348e-02, 2.72684067e-01, -6.15609705e-01,
                       1.28058004e+00, 8.69318664e-01, 4.47578311e-01,
                       -2.92379737e-01, 1.52644467e+00, 5.65292597e-01,
                       -5.93343616e-01, 2.30955884e-01, -5.11336505e-01,
                       1.23501289e+00],
                      [5.43159068e-01, 2.71917135e-01, 3.28395128e-01,
                       2.65570939e-01, 1.32801786e-01, 6.30557120e-01,
                       -1.90752923e-01, -3.81915778e-01, -3.32792163e-01,
                       -5.02242565e-01, 5.61964095e-01, 6.13379538e-01,
                       6.51038885e-02, -1.20248292e-02, -4.32104558e-01,
                       7.42150664e-01],
                      [-1.56676912e+00, 3.15348096e-02, -1.99897742e+00,
                       5.60711920e-01, 2.28923887e-01, -6.03937387e-01,
                       1.41203225e+00, 4.10895079e-01, 6.07932806e-01,
                       -7.63890073e-02, -5.03750563e-01, 5.43722093e-01,
                       -1.48856783e+00, -1.02070796e+00, -2.02205753e+00,
                       -1.85519838e+00],
                      [7.00603053e-02, 2.00205684e-01, 8.95272434e-01,
                       -5.06834447e-01, 5.52980006e-01, -6.22991025e-01,
                       8.40598702e-01, 3.23042832e-02, -2.68593878e-01,
                       -8.57947022e-02, 4.98554498e-01, -4.20765013e-01,
                       -7.43809342e-01, 3.75261515e-01, -8.94024551e-01,
                       8.47998381e-01],
                      [-3.45601201e-01, 2.11825758e-01, 1.45749485e+00,
                       -5.83331585e-01, 3.07620764e-01, -1.59492880e-01,
                       7.00160384e-01, -8.51647317e-01, 2.40840971e-01,
                       1.49827063e-01, -1.46554962e-01, -6.66770756e-01,
                       -3.17725718e-01, 8.12010288e-01, -3.09988588e-01,
                       3.65694344e-01],
                      [-1.01320839e+00, -8.58233809e-01, 3.23488176e-01,
                       5.22954613e-02, -1.80098093e+00, -2.21537709e+00,
                       1.11534703e+00, 2.56884575e+00, 1.40565410e-01,
                       -5.27087092e-01, 1.01892889e+00, -3.67924094e-01,
                       -4.48195077e-02, -1.36158693e+00, 8.56052991e-03,
                       9.03632998e-01]]]])
    parameters = {}
    parameters["W1"] = w1
    parameters["W2"] = w2
    return parameters
