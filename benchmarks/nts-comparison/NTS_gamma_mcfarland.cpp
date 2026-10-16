#include <iostream>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <chrono>

/* define costants for the ziggurat algorithm with 256 layers
r is the last right edge, a quantile
v is the area of each rectangle which must be the same*/
#define r 3.6541528853610088
#define v 0.00492867323399
#define pi 3.1415926535897932

double x[256];
double w[256];
uint32_t k[256];
double f[256];

struct XorwowState {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t a;
    uint32_t b;
    uint32_t d;
};

struct BitPool {
    uint32_t bits;
    int remaining = 0;
};

struct Devroye {
    double alpha;
    double one_minus_alpha;
    double lambda;
    double inv_alpha;
    double inv_one_minus_alpha;
    double neg_inv_one_minus_alpha;
    double lambda_alpha;
    double gamma;
    double gamma_sqrt;
    double gamma_power;
    double eps;
    double phi;
    double w1;
    double w2;
    double w3;
    double prob_w1_w2;
    double prob_w2_w3;
    double b;
    double b_lambda;
    double inv_B0;
    double meanT;
    double inv_meanT;
    double sqrt_pi_over_2;
    double rho_gamma_coeff;
    double mix12_coeff;
    bool gamma_ge_one;
};

struct GammaParameters {
    double shape, d, c;
};

GammaParameters make_gamma_parameters(double shape) {
    if (!(shape > 0.0) || !std::isfinite(shape))
        throw std::invalid_argument("Gamma shape must be finite and positive");
    const double d = (shape < 1.0 ? shape + 1.0 : shape) - 1.0 / 3.0;
    return {shape, d, 1.0 / std::sqrt(9.0 * d)};
}

struct Qu {
    double alpha;
    double lambda;
    double lambda_alpha;
    double x;
    double y;
    double logC1, logC2, logC3, logC4;
    double complement, power, log_lambda, sigma, precision;
    double log_x_factor, log_z_factor, log_normal_adjustment;
    double inv_meanT;
    int case_id;
    GammaParameters gamma;
};

/* Function needed to generate a 64 bits unsigned integer used to initialize Xorwow struct
given xorwow needs 32 bits for a state, then one generation from splitmix64 can be used to initialize 2 states in xorwow */
uint64_t splitmix64(uint64_t& state) {
    uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}


/* function used to initialize the xorwow struct using the splitmix64
the original seed is not modified because seed in init xorwowis passed by value and not reference
so a copy is created and that copy is modified internally by splitmix64 otherwise it would generate alwyas thensame value */
void init_xorwow(XorwowState& g, uint64_t seed) {
    uint64_t sm_state = seed;

    uint64_t r1 = splitmix64(sm_state);
    g.x = static_cast<uint32_t>(r1);
    g.y = static_cast<uint32_t>(r1 >> 32);

    uint64_t r2 = splitmix64(sm_state);
    g.z = static_cast<uint32_t>(r2);
    g.a = static_cast<uint32_t>(r2 >> 32);

    uint64_t r3 = splitmix64(sm_state);
    g.b = static_cast<uint32_t>(r3);
    g.d = static_cast<uint32_t>(r3 >> 32);
}


/*logic of xorwow algorithm, right shift and XOR operation to create a temporary value, this value is mixed with
actual state to create a new state via left shifts and XOR operations, then Weyl costant is updated
and states are moved left, the value returned is the new state plus Weyl costant*/
uint32_t xorwow(XorwowState& s) {
    uint32_t t = s.x ^ (s.x >> 2);
    uint32_t new_state = s.b ^ (s.b << 4) ^ t ^ (t << 1);

    s.d += 362437;
    s.x = s.y;
    s.y = s.z;
    s.z = s.a;
    s.a = s.b;
    s.b = new_state;

    return new_state + s.d;
}

/*given i need a random sign, this function takes as input a XORWOW generation and keeps 1 bit for the sign
the other 31 bits are kept to generate signs later*/
uint32_t random_bit(XorwowState& s, BitPool& pool) {
    if (pool.remaining == 0) {
        pool.bits = xorwow(s);
        pool.remaining = 32;
    }

    uint32_t bit = pool.bits & 1u;
    pool.bits >>= 1;
    pool.remaining--;

    return bit;
}


/*computes the xi recurseviley based on xi = f^-1(v/xi+1 + f(x+1))*/
void compute_xi() {
    x[255] = r;
    x[0] = 0;

    for (int i = 254; i > 0; i--) {
        x[i] = sqrt(-2 * log(v / x[i + 1] + exp(-(x[i + 1] * x[i + 1]) / 2)));
    }
}


/* computes the w, k and f according to the paper baesed on the xi*/
void compute_wki() {
    compute_xi();

    for (int i = 0; i < 256; i++) {
        if (i == 0) {
            k[i] = static_cast<uint32_t>((4294967296.0 * r * exp(-r * r / 2)) / v);
            w[i] = (1 / 4294967296.0) * (v / exp(-r * r / 2));
            f[i] = 1.0;
        } else {
            k[i] = static_cast<uint32_t>(4294967296.0 * (x[i - 1] / x[i]));
            w[i] = (1 / 4294967296.0) * x[i];
            f[i] = exp(-(x[i] * x[i]) / 2);
        }
    }
}


// McFarland (2014), arXiv:1403.6870: equal-area rectangles entirely
// under the normal density; alias-selected residual regions. This adaptation
// uses ordinary rejection in the residual wedges (no triangle squeezes).
static constexpr unsigned mc_layers = 253;
static constexpr unsigned mc_regions = 254;
static constexpr double mc_x[] = {3.6360066255009453,3.4315504938371104,3.3044597575834196,3.2104230299359235,3.1350404965487719,3.0717313805037039,3.0169132333162585,2.9684110536981407,2.9248012360226472,2.8850991264225927,2.8485942418840642,2.8147564539689531,2.7831792868860008,2.7535439605572698,2.7255956695841137,2.6991274073362836,2.6739686200329098,2.6499770531137186,2.627032766573997,2.6050336599534889,2.5838920706554447,2.5635321499209951,2.5438878118259769,2.5249011109756605,2.5065209453594202,2.4887020089326768,2.4714039381921404,2.4545906110325415,2.4382295662970614,2.4222915198391712,2.4067499583960679,2.3915807966781832,2.3767620861836303,2.3622737666172702,2.3480974526209701,2.3342162499412265,2.3206145962721516,2.3072781228890804,2.2941935338850001,2.2813485003794236,2.2687315675178263,2.2563320724428362,2.2441400717138382,2.2321462768934364,2.2203419972180249,2.2087190884339885,2.1972699070173798,2.1859872691085807,2.174864413588562,2.1638949688033051,2.1530729225103586,2.1423925946785891,2.1318486128206668,2.121435889579141,2.1111496023223131,2.1009851745363894,2.0909382588264842,2.081004721361511,2.0711806276174469,2.061462229290334,2.0518459522650252,2.0423283855384726,2.0329062710075165,2.0235764940409142,2.0143360747639161,2.0051821599912456,1.9961120157509979,1.9871230203478176,1.9782126579189359,1.9693785124412337,1.9606182621515917,1.9519296743464203,1.9433106005295073,1.9347589718802114,1.9262727950166125,1.9178501480305448,1.9094891767735223,1.9011880913744168,1.8929451629714404,1.8847587206424818,1.8766271485192187,1.8685488830716457,1.8605224105507867,1.8525462645783493,1.8446190238730034,1.8367393101037917,1.8289057858619304,1.8211171527429442,1.8133721495317059,1.8056695504835116,1.7980081636948508,1.7903868295579972,1.7828044192939809,1.7752598335589012,1.7677520011188972,1.7602798775894333,1.7528424442348562,1.7454387068244621,1.7380676945415701,1.7307284589423393,1.7234200729612774,1.7161416299605952,1.7088922428207449,1.7016710430696524,1.6944771800483069,1.6873098201105234,1.6801681458548203,1.6730513553864839,1.6659586616080069,1.658889291536187,1.6518424856442793,1.6448174972276783,1.6378135917916943,1.6308300464600651,1.6238661494029172,1.6169211992829489,1.6099945047186841,1.6030853837636858,1.5961931634006787,1.5893171790495817,1.5824567740884836,1.5756112993866491,1.5687801128486658,1.5619625789688927,1.5551580683953827,1.5483659575024959,1.5415856279714375,1.5348164663779778,1.5280578637866307,1.5213092153505889,1.5145699199167266,1.5078393796349916,1.5011169995715217,1.4944021873248317,1.4876943526444171,1.4809929070511312,1.4742972634586922,1.4676068357956762,1.4609210386273528,1.4542392867767107,1.4475609949440238,1.4408855773242908,1.4342124472218734,1.4275410166616505,1.4208706959959807,1.414200893506758,1.4075310150018168,1.4008604634049231,1.3941886383385618,1.3875149356987038,1.3808387472206944,1.3741594600353841,1.3674764562145656,1.3607891123047549,1.3540967988482895,1.34739887989068,1.3406947124730819,1.3339836461086985,1.3272650222418547,1.3205381736884116,1.3138024240561013,1.3070570871432809,1.3003014663145094,1.2935348538512339,1.2867565302757722,1.2799657636466413,1.2731618088231533,1.2663439066970441,1.2595112833887434,1.252663149405711,1.2457986987600762,1.238917108042596,1.2320175354497223,1.2250991197603072,1.2181609792581987,1.2112022105966673,1.2042218876002697,1.1972190599993775,1.1901927520921962,1.1831419613286402,1.176065656809933,1.1689627776972498,1.1618322315221077,1.1546728923905332,1.14748359907228,1.1402631529655456,1.1330103159266911,1.1257238079534488,1.118402304708936,1.1110444348725022,1.1036487773020016,1.0962138579904428,1.0887381467981605,1.0812200539395964,1.0736579262014647,1.0660500428664643,1.0583946113137479,1.0506897622639961,1.0429335446331391,1.0351239199544264,1.0272587563235849,1.019335821816135,1.0113527773194289,1.0033071687144943,0.99519641833413641,0.98701781561378432,0.97876850684000472,0.97044548388816088,0.96204557182502626,0.95356541523381555,0.94500146309757183,0.93634995205149485,0.92760688778482159,0.91876802433730931,0.9098288409930162,0.90078451642340074,0.89162989967090911,0.88235947749076726,0.87296733747965627,0.86344712631144338,0.85379200226722163,0.8439945810831877,0.8340468739370146,0.82394021614043389,0.81366518478828365,0.80321150321303558,0.7925679295829009,0.78172212632581939,0.77066050621273052,0.75936804982454387,0.7478280876642831,0.73602203822553691,0.72392909069845901,0.71152581740437215,0.69878569608271346,0.68567851518005529,0.67216962534127855,0.65821898585505023,0.64377993341176376,0.62879756815567722,0.61320660180734965,0.5969284326076898,0.57986708022339695,0.56190338937081763,0.54288651202977878,0.52262093319913649,0.50084582939368816,0.47720040650615314,0.45116153577022988,0.42192077318832577,0.38810845095539837,0.34703847379448871,0.29172225078070846};
static constexpr double mc_f[] = {0.0013464657392405521,0.0027731556153421035,0.0042547165082092052,0.0057796739057975203,0.0073412991397548456,0.0089351098420714873,0.010557880525306259,0.012207166375656877,0.013881043659552686,0.015577955319901504,0.01729661299213307,0.019035931625539331,0.020794984077516399,0.022572968529196543,0.024369184454956276,0.026183014485944015,0.028013910447927452,0.029861382426273921,0.03172499007182053,0.033604335595891849,0.035499058059143516,0.037408828665702686,0.039333346848513001,0.041272336984644283,0.043225545617491645,0.04519273909076256,0.047173701519943531,0.04916823304258678,0.051176148300674622,0.053197275117488847,0.055231453338541486,0.057278533811711732,0.059338377486157869,0.061410854613099586,0.06349584403439644,0.065593232547141098,0.067702914334350994,0.069824790453367966,0.071958768374837515,0.074104761566177832,0.07626268911432095,0.078432475383231365,0.080614049702320029,0.082807346082386213,0.085012302956157401,0.087228862940867791,0.089456972620635969,0.091696582346672953,0.0939476460535872,0.09621012109025498,0.098483968063900612,0.10076915069618246,0.10306563569021351,0.10537339260756247,0.10769239375437985,0.11002261407588633,0.11236403105853629,0.11471662463923965,0.11708037712108882,0.11945527309508601,0.12184129936742109,0.12423844489188962,0.12664670070707901,0.12906605987798558,0.13149651744175744,0.13393807035728417,0.13639071745837719,0.1388544594103131,0.14132929866952551,0.1438152394462531,0.14631228766996807,0.14882045095742252,0.15133973858316521,0.15387016145239268,0.15641173207601058,0.15896446454779223,0.16152837452352775,0.16410347920207088,0.16668979730819267,0.16928734907716389,0.17189615624099056,0.17451624201623703,0.17714763109337228,0.1797903496275855,0.18244442523101795,0.18510988696636463,0.18778676534180341,0.19047509230721302,0.19317490125164455,0.19588622700201652,0.19860910582300478,0.20134357541810191,0.20408967493182628,0.20684744495305976,0.20961692751949995,0.21239816612321172,0.21519120571726705,0.21799609272346601,0.22081287504113056,0.22364160205696795,0.22648232465600124,0.22933509523356779,0.23219996770838663,0.23507699753670006,0.23796624172749473,0.2408677588588109,0.24378160909515001,0.24670785420599284,0.24964655758544116,0.25259778427300089,0.25556160097552338,0.25853807609032509,0.26152727972950895,0.26452928374551021,0.26754416175789358,0.27057198918143149,0.27361284325549201,0.27666680307477148,0.27973394962140768,0.28281436579850855,0.28590813646514096,0.28901534847281968,0.2921360907035449,0.29527045410943437,0.29841853175400457,0.30158041885515569,0.30475621282991605,0.30794601334101046,0.31114992234531769,0.31436804414428432,0.31760048543636937,0.3208473553715967,0.32410876560829793,0.32738483037213012,0.33067566651746211,0.33398139359122425,0.33730213389932412,0.34063801257573723,0.34398915765438465,0.3473557001439222,0.35073777410556417,0.35413551673408011,0.35754906844210776,0.36097857294793106,0.36442417736688698,0.36788603230656886,0.37136429196600695,0.3748591142390183,0.37837066082192761,0.3818990973258729,0.38544459339392773,0.38900732282327832,0.39258746369271691,0.39618519849572159,0.39980071427941799,0.40343420278972908,0.40708586062304597,0.4107558893847702,0.41444449585510379,0.41815189216248533,0.42187829596510118,0.42562393064092768,0.42938902548679059,0.43317381592696519,0.43697854373187428,0.44080345724748321,0.44464881163603021,0.44851486912878352,0.45240189929155933,0.45631017930379675,0.46023999425203943,0.46419163743874431,0.4681654107074027,0.47216162478504109,0.47618059964324899,0.48022266487897619,0.48428816011644005,0.48837743543159512,0.49249085180073582,0.49662878157493712,0.50079160898218278,0.50497973065919022,0.50919355621511819,0.51343350882953609,0.51770002588724995,0.52199355965281646,0.52631457798783887,0.5306635651144318,0.53504102242856233,0.53944746936733889,0.54388344433471669,0.54834950569053964,0.55284623280833789,0.55737422720785934,0.56193411376894453,0.56652654203406116,0.57115218760760755,0.57581175366099657,0.5805059725535443,0.58523560758033821,0.59000145485956701,0.59480434537328009,0.59964514717724104,0.60452476779747577,0.6094441568333443,0.61440430878951169,0.619406266162144,0.62445112280805626,0.62954002762947892,0.63467418861169966,0.63985487725618195,0.645083433458019,0.65036127088393547,0.65568988291570485,0.66107084923410075,0.66650584313067385,0.67199663964915712,0.67754512467570327,0.68315330511808181,0.6888233203392834,0.69455745504174971,0.70035815383605404,0.70622803777408594,0.71216992318395433,0.71818684321496506,0.72428207259017252,0.7304591561765158,0.73672194212566033,0.74307462052220619,0.7495217687134158,0.75606840480484661,0.76272005121572273,0.76948281073428981,0.77636345825140762,0.78336955236056516,0.79050957241442832,0.79779308860485865,0.80523097546940869,0.8128356833747068,0.82062158872298663,0.82860545310810418,0.83680703655002786,0.84524993409648386,0.85396274566071129,0.86298075999778534,0.87234846427040236,0.8821234450360711,0.8923827790226726,0.90323423235087041,0.91483773390850387,0.92745214190924918,0.94155939144662437,0.95834165037883334};
static constexpr double mc_prob[] = {1,0.75067213202537486,0.7311484812164134,0.84203487862863946,0.98140292055252676,0.8469132492811835,0.75013127230855281,0.67684335912117022,0.61924693244203077,0.57268418476053229,0.53419427954789234,0.50180064121082391,0.47413088447467922,0.45020089302437882,0.4292856814291271,0.41083879408736634,0.39444015396871973,0.37976126307863528,0.36654135324585091,0.35457065536075161,0.34367841797887438,0.33372416852156511,0.32459123397992307,0.31618186551901067,0.3084135205402827,0.30121599268315141,0.29452917164389369,0.2883012766843201,0.28248745052263691,0.27704863030478083,0.27195063384108481,0.26716341439128122,0.26266044878328754,0.25841823167161077,0.25441585499624181,0.25063465627520032,0.24705792290760936,0.24367064231162616,0.24045928970958855,0.23741164720584498,0.23451664877752781,0.23176424696708223,0.22914529782120646,0.22665146115890175,0.22427511389726534,0.22200927443027832,0.21984753643734881,0.21778401082297494,0.21581327457535737,0.21393032565419845,0.21213054305055629,0.21040965136594458,0.20876368929874703,0.20718898158038476,0.20568211385613272,0.20423991024326138,0.20285941319050516,0.20153786533861279,0.20027269322919625,0.19906149258157294,0.19790201499679499,0.19679215590507787,0.19572994366079308,0.19471352960315647,0.1937411790557062,0.19281126309593566,0.19192225107715291,0.19107270373477567,0.19026126697674139,0.18948666611906992,0.18874770062431043,0.1880432392823812,0.18737221577860932,0.18673362459067297,0.18612651725319854,0.18554999888186624,0.18500322497415783,0.18448539846701034,0.18399576699434159,0.18353362038363916,0.18309828829546229,0.18268913808047824,0.18230557276939169,0.1819470292107595,0.1816129763329162,0.18130291357963715,0.18101636938369972,0.18075289980501724,0.1805120872476772,0.18029353926340214,0.18009688743862443,0.17992178638777684,0.17976791275606632,0.17963496438115889,0.17952265941806142,0.1794307356086193,0.17935894955871631,0.17930707608689753,0.17927490759161824,0.17926225353143987,0.17926893986547909,0.17929480858667393,0.17933971730511966,0.17940353880552987,0.17948616071489909,0.17958748514547904,0.17970742840646634,0.17984592072003666,0.18000290599306631,0.18017834158648402,0.18037219813298658,0.1805844593818107,0.18081512207191522,0.18106419577526045,0.18133170287263195,0.18161767845117879,0.18192217025437732,0.18224523871412351,0.18258695689975557,0.18294741058526148,0.18332669828622042,0.18372493132200426,0.18414223393217799,0.18457874338380617,0.18503461010366737,0.18550999786823336,0.18600508395027981,0.18652005937350447,0.18705512912152303,0.18761051240781748,0.18818644295850959,0.18878316932353695,0.18940095522873429,0.19004007993233954,0.19070083863018289,0.1913835428704905,0.19208852104497143,0.19281611884454902,0.19356669981884791,0.19434064592198583,0.19513835812777075,0.19596025705945563,0.19680678369872318,0.19767840008396084,0.19857559011559114,0.19949886035872427,0.20044874096205384,0.20142578653075488,0.20243057718148988,0.20346371956829246,0.2045258480034835,0.20561762566556549,0.20673974585720922,0.20789293337452605,0.20907794591364515,0.21029557561443638,0.21154665069916834,0.21283203717597432,0.21415264069409601,0.21550940851000791,0.21690333157872435,0.21833544677104721,0.21980683926375658,0.221318645091577,0.22287205382160416,0.2244683114849374,0.22610872366255041,0.22779465880498251,0.22952755175096073,0.23130890758375564,0.23314030564842744,0.23502340395601384,0.23695994386944083,0.23895175513332517,0.24100076132803802,0.24310898565735473,0.24527855727624162,0.24751171804694672,0.24981082985102962,0.25217838249999142,0.2546170022571112,0.25712946110039042,0.2597186867124972,0.26238777334687852,0.26513999364300833,0.26797881138838625,0.27090789550868821,0.27393113523206064,0.27705265668148038,0.2802768409576763,0.28360834393900985,0.28705211797341701,0.29061343563209979,0.29429791582726328,0.29811155254696414,0.30206074647361431,0.30615233987488033,0.3103936552294162,0.31479253784545946,0.31935740333227391,0.32409729009182481,0.32902191797257996,0.33414175353770387,0.3394680830798929,0.34501309434483585,0.35078996832093851,0.35681298237849302,0.36309762673847679,0.36966073594351889,0.37652063801796165,0.38369732384718164,0.39121264024928193,0.39909051063622925,0.40735718795099146,0.41604154559895057,0.42517541312133444,0.43479396497595518,0.44493617243073946,0.45564533080181219,0.46696967729873085,0.47896311811635484,0.49168608790752355,0.50520657094842969,0.51960132033294648,0.53495732195099444,0.55137356225896084,0.56896317646328876,0.58785607612297563,0.60820218610474253,0.63017546258879698,0.65397892170924388,0.67985098871703509,0.70807359164959416,0.7389825865706986,0.77298133903395017,0.81055863832340724,0.85231265113310573,0.89898343730331032,0.95149783339028304,0.65929343725041878,0.64826085818887769,0.56915586704938248,0.41144951069790514,0.8073752470128599,0.44922120870987636,0.61672788546191915,0.63399799633708653,0.459259881754289,0.72417303648202735,0.667229769305967,0.85901382308528351,0.32622353559228134,0.74096665379790205,0.46222090507149827};
static constexpr unsigned mc_alias[] = {0,0,1,2,0,0,0,0,0,0,0,0,0,0,0,0,1,1,2,242,244,245,246,247,247,248,248,249,249,249,250,250,250,251,251,251,251,251,251,252,252,252,252,252,252,252,252,252,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,253,3,239,240,241,242,243,244,245,246,247,248,249,250,251,252};

inline double mc_uniform(XorwowState& state) {
    return (static_cast<double>(xorwow(state)) + 0.5) * 0x1.0p-32;
}

double ziggurat(XorwowState& state, BitPool&) {
    const uint64_t bits = (uint64_t(xorwow(state)) << 32) | xorwow(state);
    const unsigned layer = bits & 255u;
    const double sign = bits & 256u ? -1.0 : 1.0;
    // Position uses disjoint bits from the layer index and sign.
    const double u = static_cast<double>(bits >> 11) * 0x1.0p-53;
    if (layer < mc_layers) return sign * u * mc_x[layer];
    const double selection = mc_uniform(state) * mc_regions;
    const unsigned slot = static_cast<unsigned>(selection);
    const unsigned region = selection-slot < mc_prob[slot] ? slot : mc_alias[slot];
    if (region == 0) {
        double tail, vertical;
        do {
            tail = -std::log(mc_uniform(state)) / mc_x[0];
            vertical = -std::log(mc_uniform(state));
        } while (2*vertical < tail*tail);
        return sign * (mc_x[0]+tail);
    }
    const double left = region == mc_layers ? 0.0 : mc_x[region];
    const double right = mc_x[region-1];
    const double bottom = mc_f[region-1];
    const double top = region == mc_layers ? 1.0 : mc_f[region];
    for (;;) {
        const double candidate = left + mc_uniform(state)*(right-left);
        const double height = bottom + mc_uniform(state)*(top-bottom);
        if (height < std::exp(-0.5*candidate*candidate)) return sign*candidate;
    }
}

/* computes values which can be precomputed for Devroje algorithm sued to generate the tilted stable */
void initDevroye(Devroye& p, double lambda, double alpha) {
    p.alpha = alpha;
    p.one_minus_alpha = 1.0 - alpha;
    p.lambda = lambda;

    p.inv_alpha = 1.0 / alpha;
    p.inv_one_minus_alpha = 1.0 / p.one_minus_alpha;
    p.neg_inv_one_minus_alpha = -p.inv_one_minus_alpha;

    p.lambda_alpha = std::pow(lambda, alpha);
    p.gamma = p.lambda_alpha * alpha * p.one_minus_alpha;
    p.gamma_sqrt = std::sqrt(p.gamma);
    p.gamma_power = std::pow(p.gamma_sqrt, p.inv_alpha);

    p.sqrt_pi_over_2 = std::sqrt(pi / 2.0);

    p.eps = ((2.0 + p.sqrt_pi_over_2) * std::sqrt(2.0 * p.gamma) + 1.0) / pi;
    p.phi = (1.0 / pi) * std::exp(-(p.gamma * pi * pi) / 8.0) * (2.0 + p.sqrt_pi_over_2) * std::sqrt(p.gamma * pi);

    p.w1 = p.eps * std::sqrt(pi / (2.0 * p.gamma));
    p.w2 = 2.0 * p.phi * std::sqrt(pi);
    p.w3 = p.eps * pi;

    p.prob_w1_w2 = p.w1 / (p.w1 + p.w2);
    p.prob_w2_w3 = p.w3 / (p.w2 + p.w3);

    p.b = p.one_minus_alpha / alpha;
    p.b_lambda = p.b * lambda;

    p.inv_B0 = std::pow(alpha, alpha) * std::pow(p.one_minus_alpha, p.one_minus_alpha);

    p.meanT = alpha * std::pow(lambda, alpha - 1.0);
    p.inv_meanT = 1.0 / p.meanT;

    p.rho_gamma_coeff = (1.0 + p.sqrt_pi_over_2) * p.gamma_sqrt;
    p.mix12_coeff = p.sqrt_pi_over_2 + 1.0;

    p.gamma_ge_one = p.gamma >= 1.0;
}

/* computes values which can be precomputed for Qu algorithm sued to generate the tilted stable */
void initQu(Qu& p, double lambda, double a) {
    p.alpha = a;
    p.lambda = lambda;
    p.lambda_alpha = pow(lambda, a);
    p.x = a * p.lambda_alpha;
    p.y = (1.0 - a) * p.lambda_alpha;

    p.complement = 1.0 - a;
    p.power = a / p.complement;
    p.log_lambda = std::log(lambda);
    p.precision = a * p.complement * p.lambda_alpha;
    p.sigma = 1.0 / std::sqrt(p.precision);
    const double log_normal_denominator = 0.5 * std::log(2.0 * pi * p.precision);
    const double logR = std::log(std::erf(std::sqrt(p.precision * pi * pi / 2.0)));

    // Compute the logs directly: do not form potentially overflowing C values.
    p.logC1 = std::lgamma(p.x) + p.x - 1.0 - p.lambda_alpha * std::log(p.x) + (p.y + 1.0) * std::log(p.power + p.x);
    p.logC2 = std::lgamma(p.y + 1.0) + p.y - p.y * std::log(p.y);
    p.logC3 = std::lgamma(p.x + 1.0) + p.x - 1.0 - p.x * std::log(p.x) - log_normal_denominator + (p.y + 1.0) * std::log1p(1.0 / p.y);
    p.logC4 = p.logC2 - log_normal_denominator;

    // Parameter-only parts of the acceptance ratios.
    p.log_x_factor = std::log(a) - std::log(p.complement)
        + p.lambda_alpha + std::lgamma(p.x) + p.power * p.log_lambda;
    p.log_z_factor = p.lambda_alpha + std::lgamma(p.y + 1.0);
    p.log_normal_adjustment = logR - log_normal_denominator;

    const double minLogC = std::min({p.logC1, p.logC2, p.logC3, p.logC4});
    if (minLogC == p.logC1) p.case_id = 0;
    else if (minLogC == p.logC2) p.case_id = 1;
    else if (minLogC == p.logC3) p.case_id = 2;
    else p.case_id = 3;
    p.gamma = make_gamma_parameters(
        p.case_id == 0 || p.case_id == 2 ? p.x : p.y + 1.0);
    p.inv_meanT = 1.0 / (a * std::pow(lambda, a - 1.0));
}

/*inline function to generate a unifomr distribution using xorwow
any uniform(a,b) can be generated as uniform01(s)*(b-a)+a. */
inline double uniform01(XorwowState& s) {
    constexpr double INV_2_32 = 1.0 / 4294967296.0;
    return (static_cast<double>(xorwow(s)) + 0.5) * INV_2_32;
}

// Marsaglia–Tsang unit-scale gamma sampler. Normal proposals use the
// existing Ziggurat; all uniform draws use the existing XORWOW stream.
double gamma_ziggurat(XorwowState& state, BitPool& pool, const GammaParameters& p) {
    for (;;) {
        const double normal = ziggurat(state, pool);
        const double base = 1.0 + p.c * normal;
        if (base <= 0.0) continue;
        const double cube = base * base * base;
        const double u = uniform01(state);
        const double square = normal * normal;
        if (u >= 1.0 - 0.0331 * square * square
            && std::log(u) >= 0.5 * square + p.d * (1.0 - cube + std::log(cube)))
            continue;
        double result = p.d * cube;
        // Gamma(k) = Gamma(k+1) * U^(1/k) for 0 < k < 1.
        if (p.shape < 1.0)
            result *= std::exp(std::log(uniform01(state)) / p.shape);
        if (result > 0.0 && std::isfinite(result)) return result;
    }
}

/* Qu algorithm used to generate the T in the NTS distribution as discussed in the paper RandomVariate Generationfor Exponential and Gamma
Tilted Stable Distributions Qu 2021. Algorithm is based on 2 dimensional Single Rejection*/
// Only the Qu arithmetic is changed; the proposal generators are retained.
double tilted_tempered_stable_Qu(XorwowState& s, BitPool& h, Qu& p) {
    for (;;) {
        double U;
        if (p.case_id < 2) {
            U = uniform01(s) * pi;
        } else {
            do {
                U = std::abs(ziggurat(s, h)) * p.sigma;
            } while (U > pi);
        }
        const double proposal = gamma_ziggurat(s, h, p.gamma);
        const double V = uniform01(s);
        // Logs require positive arguments. Zero can occur in finite precision.
        if (!(U > 0.0 && U < pi) || !(proposal > 0.0)) continue;
        const double logBU = p.alpha * std::log(std::sin(p.alpha * U))
            + p.complement * std::log(std::sin(p.complement * U))
            - std::log(std::sin(U));
        const double logProposal = std::log(proposal);
        double S, logAcceptance;

        if (p.case_id == 0 || p.case_id == 2) {
            // X proposal: log(x1*x2*x3*x4*x5/C).
            S = proposal / p.lambda;
            const double logx2 = logBU / p.complement;
            const double logx5 = -std::exp(
                logx2 + p.power * p.log_lambda - p.power * logProposal);
            logAcceptance = p.log_x_factor + logx2
                - (p.power + p.x) * logProposal + logx5
                - (p.case_id == 0 ? p.logC1 : p.logC3);
        } else {
            // Z proposal: compute S from its log, and acceptance directly in logs.
            S = std::exp((logBU - p.complement * logProposal) / p.alpha);
            logAcceptance = p.log_z_factor - p.y * logProposal - p.lambda * S
                - (p.case_id == 1 ? p.logC2 : p.logC4);
        }
        if (p.case_id >= 2) {
            logAcceptance += p.log_normal_adjustment + 0.5 * p.precision * U * U;
        }
        if (S > 0.0 && std::isfinite(S) && std::log(V) <= logAcceptance) return S;
    }
}


/* Devroje algorithm used to generate the T in the NTS distribution as discussed in the paper RANDOM VARIATE GENERATION 
FOR EXPONENTIALLY AND POLYNOMIALLY TILTED STABLE DISTRIBUTIONS Devroje . Algorithm is based on Double Rejection algorithm.
This is the main reason Qu is faster*/
double tilted_tempered_stable_Devroye(XorwowState& s, BitPool& h, const Devroye& p) {
    double U;
    double z;
    double Z;
    double BU;

    for (;;) {

        // First rejection stage
        for (;;) {
            double V = uniform01(s);
            double W1 = uniform01(s);

            if (p.gamma_ge_one) {
                if (V < p.prob_w1_w2) U = std::abs(ziggurat(s, h)) / p.gamma_sqrt;
                else U = pi * (1.0 - W1 * W1);
            } else {
                if (V < p.prob_w2_w3) U = pi * W1;
                else U = pi * (1.0 - W1 * W1);
            }

            if (U >= pi) continue;

            double W = uniform01(s);

            BU = std::sin(U) /
                 (std::pow(std::sin(p.alpha * U), p.alpha) *
                  std::pow(std::sin(p.one_minus_alpha * U), p.one_minus_alpha));

            double c = std::sqrt(BU * p.inv_B0);
            double vi = std::pow(p.gamma_sqrt + p.alpha * c, p.inv_alpha);

            z = vi / (vi - p.gamma_power);

            double gss;

            if (p.gamma_ge_one) gss = p.eps * std::exp(-p.gamma * U * U / 2.0);
            else gss = p.eps;

            if (U > 0.0) gss += p.phi / std::sqrt(pi - U);

            double rho = pi * std::exp(-p.lambda_alpha * (1.0 - 1.0 / (c * c))) * gss /
                         (p.rho_gamma_coeff / c + z);

            Z = W * rho;

            if (Z <= 1.0) break;
        }

        double a = std::pow(BU, p.neg_inv_one_minus_alpha);
        double m = std::pow(p.b_lambda / a, p.alpha);
        double h1 = std::sqrt((m * p.alpha) / a);

        double a1 = h1 * p.sqrt_pi_over_2;
        double a3 = z / a;
        double s1 = h1 * p.mix12_coeff + a3;

        double V1 = uniform01(s);
        double E1 = 0.0;
        double N1 = 0.0;
        double X;

        if (V1 < a1 / s1) {
            N1 = ziggurat(s, h);
            X = m - h1 * std::abs(N1);
        } else if (V1 < (h1 * p.mix12_coeff) / s1) {
            X = m + h1 * uniform01(s);
        } else {
            E1 = -std::log(uniform01(s));
            X = m + h1 + a3 * E1;
        }

        if (X <= 0.0) continue;

        double E = -std::log(Z);
        double X_neg_b = std::pow(X, -p.b);
        double m_neg_b = std::pow(m, -p.b);
        double gf = a * (X - m) + p.lambda * (X_neg_b - m_neg_b);

        if (X < m) gf -= (N1 * N1) / 2.0;
        if (X > m + h1) gf -= E1;

        if (gf <= E) return X_neg_b;
    }
}

/*Union of tilted simulation Devroje and ziggurat. T is modified in order to have expected value 0 */
double NTSDevroye(XorwowState& state, BitPool& pool, const Devroye& p, double beta, double mu, double sigma) {
    double T = tilted_tempered_stable_Devroye(state, pool, p);
    double T_scaled = T * p.inv_meanT;
    double Z = ziggurat(state, pool);

    return mu + beta * (T_scaled - 1.0) + sigma * std::sqrt(T_scaled) * Z;
}

/*Union of tilted simulation Qu and ziggurat. T is modified in order to have expected value 0 */
double NTSQu(XorwowState& state, BitPool& pool, Qu& p, double beta, double mu, double sigma) {
    double T = tilted_tempered_stable_Qu(state, pool, p);
    double T_scaled = T * p.inv_meanT;
    double Z = ziggurat(state, pool);
    return mu + beta * (T_scaled - 1.0) + sigma * std::sqrt(T_scaled) * Z;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./program <seed>\n";
        return 1;
    }

    const int N = 1000000;
    std::vector<double> numbers(N);

    uint64_t seed = std::stoull(argv[1]);
    BitPool h;
    XorwowState g;
    Devroye p;
    Qu q;

    initDevroye(p, 0.5, 0.5);
    initQu(q, 0.5, 0.5);
    init_xorwow(g, seed);
    compute_wki();

    auto start4 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) numbers[i] = NTSQu(g, h, q, 0.5, 0.0, 1.0);

    auto end4 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed4 = end4 - start4;

    std::cout << "Qu -> Number of RVs generated " << N << " in " << elapsed4.count() << " ms\n";
}