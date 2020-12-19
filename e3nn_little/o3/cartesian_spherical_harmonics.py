# pylint: disable=not-callable, no-member, invalid-name, line-too-long, wildcard-import, unused-wildcard-import, missing-docstring, bare-except
import math

import torch

from e3nn_little import o3


@torch.jit.script
def y0(x, _y, _z):  # pragma: no cover
    return torch.ones(x.shape + (1,), dtype=x.dtype, device=x.device)


@torch.jit.script
def y1(x, y, z):  # pragma: no cover
    return -1.73205080756888 * torch.stack([
        y,
        z,
        x,
    ], dim=-1)


@torch.jit.script
def y2(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    return torch.stack([
        3.87298334620742*x*y,
        3.87298334620742*y*z,
        -3.35410196624968*xx - 3.35410196624968*yy + 2.23606797749979,
        3.87298334620742*x*z,
        1.93649167310371*xx - 1.93649167310371*yy,
    ], dim=-1)


@torch.jit.script
def y3(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        -6.27495019900557*xx*y + 2.09165006633519*y**3,
        -10.2469507659596*x*y*z,
        8.10092587300982*xx*y + y*(8.10092587300982*yy - 6.48074069840786),
        z*(3.96862696659689 - 6.61437827766148*zz),
        x*(8.10092587300982*xx + 8.10092587300982*yy - 6.48074069840786),
        -10.2469507659596*xx*z + z*(5.1234753829798 - 5.1234753829798*zz),
        x*(-2.09165006633519*xx + 6.27495019900557*yy),
    ], dim=-1)


@torch.jit.script
def y4(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        x*(8.87411967464942*xx*y - 8.87411967464942*y**3),
        y*(-25.0998007960223*yy*z + z*(18.8248505970167 - 18.8248505970167*zz)),
        x*(-23.4787137637478*xx*y + y*(20.1246117974981 - 23.4787137637478*yy)),
        y*z*(16.601957715884*zz - 7.11512473537885),
        xx*(13.125*xx + 26.25*yy - 15.0) + yy*(13.125*yy - 15.0) + 3.0,
        x*z*(16.601957715884*zz - 7.11512473537885),
        xx*(10.0623058987491 - 11.7393568818739*xx) + yy*(11.7393568818739*yy - 10.0623058987491),
        x*(25.0998007960223*xx*z + z*(18.8248505970167*zz - 18.8248505970167)),
        xx*(2.21852991866236*xx - 13.3111795119741*yy) + 2.21852991866236*y**4,
    ], dim=-1)


@torch.jit.script
def y5(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        xx*(-11.6340690431164*xx*y + 23.2681380862329*y**3) - 2.32681380862329*y**5,
        x*(-29.4321253055229*xx*y*z + 29.4321253055229*y**3*z),
        xx*(46.8262246236017*xx*y + y*(31.2174830824011*yy - 41.6233107765348)) + y**3*(13.8744369255116 - 15.6087415412006*yy),
        x*(16.9926454679664*xx*y*z + y*(16.9926454679664*yy*z - 33.9852909359329*z**3)),
        xx*(-33.718735518996*xx*y + y*(44.958314025328 - 67.4374710379919*yy)) + y*(yy*(44.958314025328 - 33.718735518996*yy) - 12.8452325786651),
        z*(zz*(29.0204669156097 - 26.1184202240488*zz) - 6.21867148191637),
        x*(xx*(-33.718735518996*xx - 67.4374710379919*yy + 44.958314025328) + yy*(44.958314025328 - 33.718735518996*yy) - 12.8452325786651),
        xx*z*(16.9926454679664 - 50.9779364038993*zz) + z*(zz*(33.9852909359329 - 25.4889682019496*zz) - 8.49632273398321),
        x*(xx*(15.6087415412006*xx - 31.2174830824011*yy - 13.8744369255116) + yy*(41.6233107765348 - 46.8262246236017*yy)),
        xx*(-58.8642506110457*xx*z + z*(58.8642506110457 - 58.8642506110457*zz)) + z*(zz*(14.7160626527614 - 7.35803132638072*zz) - 7.35803132638072),
        x*(xx*(-2.32681380862329*xx + 23.2681380862329*yy) - 11.6340690431164*y**4),
    ], dim=-1)


@torch.jit.script
def y6(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        x*(xx*(14.5309475774982*xx*y - 48.4364919249939*y**3) + 14.5309475774982*y**5),
        y*(yy*(134.231143927183*yy*z + z*(167.788929908978*zz - 167.788929908978)) + z*(zz*(41.9472324772446*zz - 83.8944649544891) + 41.9472324772446)),
        x*(xx*(-78.699984116898*xx*y + 71.5454401062709*y) + y**3*(78.699984116898*yy - 71.5454401062709)),
        y*(yy*z*(39.1870514328394 - 143.685855253744*zz) + z*(zz*(137.154680014938 - 107.764391440308*zz) - 29.3902885746295)),
        x*(xx*(107.764391440308*xx*y + y*(215.528782880617*yy - 156.748205731358)) + y*(yy*(107.764391440308*yy - 156.748205731358) + 52.2494019104525)),
        y*z*(zz*(68.1561855226655*zz - 61.9601686569686) + 10.3266947761614),
        xx*(xx*(-52.0551465395113*xx - 156.165439618534*yy + 85.1811488828367) + yy*(170.362297765673 - 156.165439618534*yy) - 37.8582883923719) + yy*(yy*(85.1811488828367 - 52.0551465395113*yy) - 37.8582883923719) + 3.60555127546399,
        x*z*(zz*(68.1561855226655*zz - 61.9601686569686) + 10.3266947761614),
        xx*(xx*(53.8821957201542*xx + 53.8821957201542*yy - 78.3741028656788) - 53.8821957201542*y**4 + 26.1247009552263) + yy*(yy*(78.3741028656788 - 53.8821957201542*yy) - 26.1247009552263),
        x*(xx*z*(143.685855253744*zz - 39.1870514328394) + z*(zz*(107.764391440308*zz - 137.154680014938) + 29.3902885746295)),
        xx*(xx*(-19.6749960292245*xx + 98.3749801461225*yy + 17.8863600265677) + yy*(98.3749801461225*yy - 107.318160159406)) + y**4*(17.8863600265677 - 19.6749960292245*yy),
        x*(xx*(134.231143927183*xx*z + z*(167.788929908978*zz - 167.788929908978)) + z*(zz*(41.9472324772446*zz - 83.8944649544891) + 41.9472324772446)),
        xx*(xx*(2.4218245962497*xx - 36.3273689437454*yy) + 36.3273689437454*y**4) - 2.4218245962497*y**6,
    ], dim=-1)


@torch.jit.script
def y7(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        xx*(xx*(-17.5477863187212*xx*y + 87.7389315936062*y**3) - 52.6433589561637*y**5) + 2.50682661696018*y**7,
        x*(xx*(-56.2781179722634*xx*y*z + 187.593726574211*y**3*z) - 56.2781179722634*y**5*z),
        xx*(xx*(119.568009053687*xx*y + y*(-119.568009053687*yy - 110.370469895711)) + y**3*(220.740939791422 - 215.222416296636*yy)) + y**5*(23.9136018107373*yy - 22.0740939791422),
        x*(xx*(44.1481879582843*xx*y*z - 147.160626527614*y*z**3) + y**3*(-44.1481879582843*yy*z + 147.160626527614*z**3)),
        xx*(xx*(-237.937333776538*xx*y + y*(366.057436579289 - 396.562222960896*yy)) + y*(yy*(244.038291052859 - 79.3124445921792*yy) - 133.111795119741)) + y**3*(yy*(79.3124445921792*yy - 122.01914552643) + 44.3705983732471),
        x*(xx*(-23.5310632462709*xx*y*z + y*(-47.0621264925418*yy*z + 125.499003980111*z**3)) + y*(yy*(-23.5310632462709*yy*z + 125.499003980111*z**3) - 75.2994023880668*z**5)),
        xx*(xx*(137.373183706146*xx*y + y*(412.119551118438*yy - 253.6120314575)) + y*(yy*(412.119551118438*yy - 507.224062915) + 138.333835340455)) + y*(yy*(yy*(137.373183706146*yy - 253.6120314575) + 138.333835340455) - 20.4939015319192),
        z*(zz*(zz*(167.748591182609 - 103.844365970186*zz) - 76.2493596284585) + 8.47215106982872),
        x*(xx*(xx*(137.373183706146*xx + 412.119551118438*yy - 253.6120314575) + yy*(412.119551118438*yy - 507.224062915) + 138.333835340455) + yy*(yy*(137.373183706146*yy - 253.6120314575) + 138.333835340455) - 20.4939015319192),
        xx*z*(zz*(172.561130472653 - 224.329469614449*zz) - 23.5310632462709) + z*(zz*(zz*(198.445300043551 - 112.164734807225*zz) - 98.046096859462) + 11.7655316231354),
        x*(xx*(xx*(-79.3124445921792*xx + 79.3124445921792*yy + 122.01914552643) + yy*(396.562222960896*yy - 244.038291052859) - 44.3705983732471) + yy*(yy*(237.937333776538*yy - 366.057436579289) + 133.111795119741)),
        xx*(xx*z*(88.2963759165686 - 382.617628971797*zz) + z*(zz*(470.914004888366 - 382.617628971797*zz) - 88.2963759165686)) + z*(zz*(zz*(106.69145423252 - 47.8272036214747*zz) - 69.9012976006168) + 11.0370469895711),
        x*(xx*(xx*(23.9136018107373*xx - 215.222416296636*yy - 22.0740939791422) + yy*(220.740939791422 - 119.568009053687*yy)) + y**4*(119.568009053687*yy - 110.370469895711)),
        xx*(xx*(-300.149962518738*xx*z + z*(450.224943778107 - 450.224943778107*zz)) + z*(zz*(337.668707833581 - 168.83435391679*zz) - 168.83435391679)) + z*(zz*(zz*(28.1390589861317 - 9.37968632871057*zz) - 28.1390589861317) + 9.37968632871057),
        x*(xx*(xx*(-2.50682661696018*xx + 52.6433589561637*yy) - 87.7389315936062*y**4) + 17.5477863187212*y**6),
    ], dim=-1)


@torch.jit.script
def y8(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        x*(xx*(xx*(20.6718218536732*xx*y - 144.702752975712*y**3) + 144.702752975712*y**5) - 20.6718218536732*y**7),
        y*(yy*(yy*(-661.498299317542*yy*z + z*(1157.6220238057 - 1157.6220238057*zz)) + z*(zz*(1157.6220238057 - 578.811011902849*zz) - 578.811011902849)) + z*(zz*(zz*(217.054129463568 - 72.3513764878561*zz) - 217.054129463568) + 72.3513764878561)),
        x*(xx*(xx*(-169.836347009776*xx*y + y*(396.284809689477*yy + 158.513923875791)) + y**3*(396.284809689477*yy - 528.379746252636)) + y**5*(158.513923875791 - 169.836347009776*yy)),
        y*(yy*(yy*z*(978.369178786822*zz - 195.673835757364) + z*(zz*(1222.96147348353*zz - 1467.55376818023) + 244.592294696705)) + z*(zz*(zz*(305.740368370882*zz - 672.62881041594) + 428.036515719235) - 61.1480736741764)),
        x*(xx*(xx*(440.945030056185*xx*y + y*(440.945030056185*yy - 705.512048089896)) + y*(271.350787726883 - 440.945030056185*y**4)) + y**3*(yy*(705.512048089896 - 440.945030056185*yy) - 271.350787726883)),
        y*(yy*z*(zz*(455.406068800142 - 683.109103200214*zz) - 52.5468540923241) + z*(zz*(zz*(853.886379000267 - 512.33182740016*zz) - 380.96469216935) + 39.4101405692431)),
        x*(xx*(xx*(-462.46704907958*xx*y + y*(924.934098159161 - 1387.40114723874*yy)) + y*(yy*(1849.86819631832 - 1387.40114723874*yy) - 569.190214251791)) + y*(yy*(yy*(924.934098159161 - 462.46704907958*yy) - 569.190214251791) + 103.489129863962)),
        y*z*(zz*(zz*(276.376923967184*zz - 386.927693554057) + 148.818343674637) - 13.5289403340579),
        xx*(xx*(xx*(207.282692975388*xx + 829.130771901551*yy - 442.203078347494) + yy*(1243.69615785233*yy - 1326.60923504248) + 306.140592702111) + yy*(yy*(829.130771901551*yy - 1326.60923504248) + 612.281185404223) - 74.2159012611179) + yy*(yy*(yy*(207.282692975388*yy - 442.203078347494) + 306.140592702111) - 74.2159012611179) + 4.12310562561766,
        x*z*(zz*(zz*(276.376923967184*zz - 386.927693554057) + 148.818343674637) - 13.5289403340579),
        xx*(xx*(xx*(-231.23352453979*xx - 462.46704907958*yy + 462.46704907958) + 462.46704907958*yy - 284.595107125896) + y**4*(462.46704907958*yy - 462.46704907958) + 51.744564931981) + yy*(yy*(yy*(231.23352453979*yy - 462.46704907958) + 284.595107125896) - 51.744564931981),
        x*(xx*z*(zz*(683.109103200214*zz - 455.406068800142) + 52.5468540923241) + z*(zz*(zz*(512.33182740016*zz - 853.886379000267) + 380.96469216935) - 39.4101405692431)),
        xx*(xx*(xx*(110.236257514046*xx - 440.945030056185*yy - 176.378012022474) + yy*(881.89006011237 - 1102.36257514046*yy) + 67.8376969317208) + yy*(yy*(881.89006011237 - 440.945030056185*yy) - 407.026181590325)) + y**4*(yy*(110.236257514046*yy - 176.378012022474) + 67.8376969317208),
        x*(xx*(xx*z*(978.369178786822*zz - 195.673835757364) + z*(zz*(1222.96147348353*zz - 1467.55376818023) + 244.592294696705)) + z*(zz*(zz*(305.740368370882*zz - 672.62881041594) + 428.036515719235) - 61.1480736741764)),
        xx*(xx*(xx*(-28.3060578349626*xx + 396.284809689477*yy + 26.4189873126318) - 396.284809689477*yy) + y**4*(396.284809689477 - 396.284809689477*yy)) + y**6*(28.3060578349626*yy - 26.4189873126318),
        x*(xx*(xx*(661.498299317542*xx*z + z*(1157.6220238057*zz - 1157.6220238057)) + z*(zz*(578.811011902849*zz - 1157.6220238057) + 578.811011902849)) + z*(zz*(zz*(72.3513764878561*zz - 217.054129463568) + 217.054129463568) - 72.3513764878561)),
        xx*(xx*(xx*(2.58397773170915*xx - 72.3513764878561*yy) + 180.87844121964*y**4) - 72.3513764878561*y**6) + 2.58397773170915*y**8,
    ], dim=-1)


@torch.jit.script
def y9(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        xx*(xx*(xx*(-23.8930627690618*xx*y + 223.00191917791*y**3) - 334.502878766865*y**5) + 95.5722510762473*y**7) - 2.65478475211798*y**9,
        x*(xx*(xx*(-90.106382439037*xx*y*z + 630.744677073259*y**3*z) - 630.744677073259*y**5*z) + 90.106382439037*y**7*z),
        xx*(xx*(xx*(229.865116871494*xx*y + y*(-919.460467485977*yy - 216.343639408465)) + y**3*(1081.71819704233 - 459.730233742989*yy)) + y**5*(656.757476775698*yy - 649.030918225396)) + y**7*(30.9062342012093 - 32.8378738387849*yy),
        x*(xx*(xx*(80.2967518606762*xx*y*z + y*(-187.359087674911*yy*z - 374.718175349822*z**3)) + y**3*(-187.359087674911*yy*z + 1249.06058449941*z**3)) + y**5*(80.2967518606762*yy*z - 374.718175349822*z**3)),
        xx*(xx*(xx*(-734.27718140085*xx*y + 1209.39771054258*y) + y*(yy*(2055.97610792238*yy - 1209.39771054258) - 483.75908421703)) + y**3*(yy*(1174.84349024136*yy - 2176.91587897664) + 967.518168434061)) + y**5*(yy*(241.879542108515 - 146.85543628017*yy) - 96.7518168434061),
        x*(xx*(xx*(-57.8202697481601*xx*y*z + y*(-57.8202697481601*yy*z + 462.562157985281*z**3)) + y*(57.8202697481601*y**4*z - 462.562157985281*z**5)) + y**3*(yy*(57.8202697481601*yy*z - 462.562157985281*z**3) + 462.562157985281*z**5)),
        xx*(xx*(xx*(1085.14144073993*xx*y + y*(2893.71050863982*yy - 2297.94658039044)) + y*(yy*(2170.28288147986*yy - 3829.9109673174) + 1531.96438692696)) + y*(yy*(1021.30959128464 - 765.982193463481*yy) - 314.249105010659)) + y**3*(yy*(yy*(765.982193463481 - 361.713813579977*yy) - 510.65479564232) + 104.74970167022),
        x*(xx*(xx*(30.001464807989*xx*y*z + y*(90.0043944239669*yy*z - 300.01464807989*z**3)) + y*(yy*(90.0043944239669*yy*z - 600.029296159779*z**3) + 480.023436927823*z**5)) + y*(yy*(yy*(30.001464807989*yy*z - 300.01464807989*z**3) + 480.023436927823*z**5) - 137.14955340795*z**7)),
        xx*(xx*(xx*(-555.338837161654*xx*y + y*(1306.67961685095 - 2221.35534864662*yy)) + y*(yy*(3920.03885055285 - 3332.03302296993*yy) - 1045.34369348076)) + y*(yy*(yy*(3920.03885055285 - 2221.35534864662*yy) - 2090.68738696152) + 321.644213378696)) + y*(yy*(yy*(yy*(1306.67961685095 - 555.338837161654*yy) - 1045.34369348076) + 321.644213378696) - 29.2403830344269),
        z*(zz*(zz*(zz*(876.547334427632 - 413.925130146382*zz) - 613.583134099343) + 157.329008743421) - 10.7269778688696),
        x*(xx*(xx*(xx*(-555.338837161654*xx - 2221.35534864662*yy + 1306.67961685095) + yy*(3920.03885055285 - 3332.03302296993*yy) - 1045.34369348076) + yy*(yy*(3920.03885055285 - 2221.35534864662*yy) - 2090.68738696152) + 321.644213378696) + yy*(yy*(yy*(1306.67961685095 - 555.338837161654*yy) - 1045.34369348076) + 321.644213378696) - 29.2403830344269),
        xx*z*(zz*(zz*(1170.05712751157 - 947.189103223651*zz) - 390.019042503856) + 30.001464807989) + z*(zz*(zz*(zz*(1058.62311536761 - 473.594551611826*zz) - 780.038085007713) + 210.010253655923) - 15.0007324039945),
        x*(xx*(xx*(xx*(361.713813579977*xx - 765.982193463481) + yy*(765.982193463481 - 2170.28288147986*yy) + 510.65479564232) + yy*(yy*(3829.9109673174 - 2893.71050863982*yy) - 1021.30959128464) - 104.74970167022) + yy*(yy*(yy*(2297.94658039044 - 1085.14144073993*yy) - 1531.96438692696) + 314.249105010659)),
        xx*(xx*z*(zz*(1156.4053949632 - 1965.88917143744*zz) - 115.64053949632) + z*(zz*(zz*(3122.29456640065 - 1965.88917143744*zz) - 1272.04593445952) + 115.64053949632)) + z*(zz*(zz*(zz*(636.022967229761 - 245.736146429681*zz) - 549.292562607521) + 173.46080924448) - 14.45506743704),
        x*(xx*(xx*(xx*(-146.85543628017*xx + 1174.84349024136*yy + 241.879542108515) + yy*(2055.97610792238*yy - 2176.91587897664) - 96.7518168434061) + yy*(967.518168434061 - 1209.39771054258*yy)) + y**4*(yy*(1209.39771054258 - 734.27718140085*yy) - 483.75908421703)),
        xx*(xx*(xx*z*(428.249343256939 - 2426.74627845599*zz) + z*(zz*(4282.49343256939 - 3640.11941768399*zz) - 642.374014885409)) + z*(zz*(zz*(2970.97981884502 - 1365.04478163149*zz) - 1846.82529279555) + 240.890255582028)) + z*(zz*(zz*(zz*(240.890255582028 - 75.8358212017497*zz) - 267.655839535587) + 115.984197132088) - 13.3827919767794),
        x*(xx*(xx*(xx*(32.8378738387849*xx - 656.757476775698*yy - 30.9062342012093) + yy*(459.730233742989*yy + 649.030918225396)) + y**4*(919.460467485977*yy - 1081.71819704233)) + y**6*(216.343639408465 - 229.865116871494*yy)),
        xx*(xx*(xx*(-1441.70211902459*xx*z + z*(2883.40423804918 - 2883.40423804918*zz)) + z*(zz*(3604.25529756148 - 1802.12764878074*zz) - 1802.12764878074)) + z*(zz*(zz*(1081.27658926844 - 360.425529756148*zz) - 1081.27658926844) + 360.425529756148)) + z*(zz*(zz*(zz*(45.0531912195185 - 11.2632978048796*zz) - 67.5797868292778) + 45.0531912195185) - 11.2632978048796),
        x*(xx*(xx*(xx*(-2.65478475211798*xx + 95.5722510762473*yy) - 334.502878766865*y**4) + 223.00191917791*y**6) - 23.8930627690618*y**8),
    ], dim=-1)


@torch.jit.script
def y10(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        x*(xx*(xx*(xx*(27.2034486491732*xx*y - 326.441383790078*y**3) + 685.526905959165*y**5) - 326.441383790078*y**7) + 27.2034486491732*y**9),
        y*(yy*(yy*(yy*(3114.43253258118*yy*z + z*(7007.47319830765*zz - 7007.47319830765)) + z*(zz*(5255.60489873073*zz - 10511.2097974615) + 5255.60489873073)) + z*(zz*(zz*(1459.89024964743*zz - 4379.67074894228) + 4379.67074894228) - 1459.89024964743)) + z*(zz*(zz*(zz*(109.491768723557*zz - 437.967074894228) + 656.950612341342) - 437.967074894228) + 109.491768723557)),
        x*(xx*(xx*(xx*(-299.978929924149*xx*y + y*(1799.87357954489*yy + 284.190565191299)) - 1989.33395633909*y**3) + y**5*(1989.33395633909 - 1799.87357954489*yy)) + y**7*(299.978929924149*yy - 284.190565191299)),
        y*(yy*(yy*(yy*z*(928.162499242455 - 5878.36249520221*zz) + z*(zz*(11911.4187402782 - 10287.1343666039*zz) - 1624.2843736743)) + z*(zz*(zz*(11099.276553441 - 5143.56718330194*zz) - 6767.85155697623) + 812.142186837148)) + z*(zz*(zz*(zz*(2030.35546709287 - 642.945897912742*zz) - 2233.39101380216) + 947.499217976673) - 101.517773354644)),
        x*(xx*(xx*(xx*(1136.11450656507*xx*y + y*(-1514.81934208676*yy - 1913.45601105696)) + y*(yy*(4464.73069246623 - 5301.86769730365*yy) + 787.893651611688)) + y**3*(yy*(4464.73069246623 - 1514.81934208676*yy) - 2626.31217203896)) + y**5*(yy*(1136.11450656507*yy - 1913.45601105696) + 787.893651611688)),
        y*(yy*(yy*z*(zz*(5419.58243606*zz - 2852.41180845263) + 251.683394863467) + z*(zz*(zz*(6774.47804507499*zz - 10339.9928056408) + 3880.11900414512) - 314.604243579334)) + z*(zz*(zz*(zz*(1693.61951126875*zz - 4278.61771267894) + 3555.02795244648) - 1048.68081193111) + 78.6510608948335)),
        x*(xx*(xx*(xx*(-2142.27805812418*xx*y + y*(4735.56202322187 - 4284.55611624836*yy)) + y*(4735.56202322187*yy - 3342.74966345073)) + y*(y**4*(4284.55611624836*yy - 4735.56202322187) + 742.833258544608)) + y**3*(yy*(yy*(2142.27805812418*yy - 4735.56202322187) + 3342.74966345073) - 742.833258544608)),
        y*(yy*z*(zz*(zz*(3348.54801934967 - 3029.63868417351*zz) - 984.86706451461) + 65.657804300974) + z*(zz*(zz*(zz*(4783.64002764239 - 2272.22901313014*zz) - 3250.06131289821) + 787.893651611688) - 49.2433532257305)),
        x*(xx*(xx*(xx*(1931.02334621704*xx*y + y*(7724.09338486816*yy - 4878.37476939042)) + y*(yy*(11586.1400773022*yy - 14635.1243081712) + 4304.44832593272)) + y*(yy*(yy*(7724.09338486816*yy - 14635.1243081712) + 8608.89665186544) - 1530.47051588719)) + y*(yy*(yy*(yy*(1931.02334621704*yy - 4878.37476939042) + 4304.44832593272) - 1530.47051588719) + 176.592751833137)),
        y*z*(zz*(zz*(zz*(1114.87684874986*zz - 2112.39823973658) + 1304.7165598373) - 289.937013297177) + 16.7271353825295),
        xx*(xx*(xx*(xx*(-826.814799899669*xx - 4134.07399949835*yy + 2175.8284207886) + yy*(8703.31368315441 - 8268.14799899669*yy) - 2047.83851368339) + yy*(yy*(13054.9705247316 - 8268.14799899669*yy) - 6143.51554105017) + 819.135405473356) + yy*(yy*(yy*(8703.31368315441 - 4134.07399949835*yy) - 6143.51554105017) + 1638.27081094671) - 126.020831611286) + yy*(yy*(yy*(yy*(2175.8284207886 - 826.814799899669*yy) - 2047.83851368339) + 819.135405473356) - 126.020831611286) + 4.58257569495584,
        x*z*(zz*(zz*(zz*(1114.87684874986*zz - 2112.39823973658) + 1304.7165598373) - 289.937013297177) + 16.7271353825295),
        xx*(xx*(xx*(xx*(965.51167310852*xx + 2896.53501932556*yy - 2439.18738469521) + yy*(1931.02334621704*yy - 4878.37476939042) + 2152.22416296636) + yy*(2152.22416296636 - 1931.02334621704*y**4) - 765.235257943595) + y**4*(yy*(4878.37476939042 - 2896.53501932556*yy) - 2152.22416296636) + 88.2963759165686) + yy*(yy*(yy*(yy*(2439.18738469521 - 965.51167310852*yy) - 2152.22416296636) + 765.235257943595) - 88.2963759165686),
        x*(xx*z*(zz*(zz*(3029.63868417351*zz - 3348.54801934967) + 984.86706451461) - 65.657804300974) + z*(zz*(zz*(zz*(2272.22901313014*zz - 4783.64002764239) + 3250.06131289821) - 787.893651611688) + 49.2433532257305)),
        xx*(xx*(xx*(xx*(-535.569514531045*xx + 1606.70854359314*yy + 1183.89050580547) + yy*(7497.97320343463*yy - 4735.56202322187) - 835.687415862684) + yy*(yy*(7497.97320343463*yy - 11838.9050580547) + 4178.43707931342) + 185.708314636152) + yy*(yy*(yy*(1606.70854359314*yy - 4735.56202322187) + 4178.43707931342) - 1114.24988781691)) + y**4*(yy*(yy*(1183.89050580547 - 535.569514531045*yy) - 835.687415862684) + 185.708314636152),
        x*(xx*(xx*z*(zz*(5419.58243606*zz - 2852.41180845263) + 251.683394863467) + z*(zz*(zz*(6774.47804507499*zz - 10339.9928056408) + 3880.11900414512) - 314.604243579334)) + z*(zz*(zz*(zz*(1693.61951126875*zz - 4278.61771267894) + 3555.02795244648) - 1048.68081193111) + 78.6510608948335)),
        xx*(xx*(xx*(xx*(189.352417760845*xx - 2461.58143089098*yy - 318.909335176159) + yy*(4464.73069246623 - 2650.93384865183*yy) + 131.315608601948) + yy*(2650.93384865183*y**4 - 1969.73412902922)) + y**4*(yy*(2461.58143089098*yy - 4464.73069246623) + 1969.73412902922)) + y**6*(yy*(318.909335176159 - 189.352417760845*yy) - 131.315608601948),
        x*(xx*(xx*(xx*z*(5878.36249520221*zz - 928.162499242455) + z*(zz*(10287.1343666039*zz - 11911.4187402782) + 1624.2843736743)) + z*(zz*(zz*(5143.56718330194*zz - 11099.276553441) + 6767.85155697623) - 812.142186837148)) + z*(zz*(zz*(zz*(642.945897912742*zz - 2030.35546709287) + 2233.39101380216) - 947.499217976673) + 101.517773354644)),
        xx*(xx*(xx*(xx*(-37.4973662405186*xx + 1012.428888494*yy + 35.5238206489124) + yy*(-1574.88938210178*yy - 994.666978169547)) + y**4*(2486.66744542387 - 1574.88938210178*yy)) + y**6*(1012.428888494*yy - 994.666978169547)) + y**8*(35.5238206489124 - 37.4973662405186*yy),
        x*(xx*(xx*(xx*(3114.43253258118*xx*z + z*(7007.47319830765*zz - 7007.47319830765)) + z*(zz*(5255.60489873073*zz - 10511.2097974615) + 5255.60489873073)) + z*(zz*(zz*(1459.89024964743*zz - 4379.67074894228) + 4379.67074894228) - 1459.89024964743)) + z*(zz*(zz*(zz*(109.491768723557*zz - 437.967074894228) + 656.950612341342) - 437.967074894228) + 109.491768723557)),
        xx*(xx*(xx*(xx*(2.72034486491732*xx - 122.415518921279*yy) + 571.272421632637*y**4) - 571.272421632637*y**6) + 122.415518921279*y**8) - 2.72034486491732*y**10,
    ], dim=-1)


@torch.jit.script
def y11(x, y, z):  # pragma: no cover
    xx = x**2
    yy = y**2
    zz = z**2
    return torch.stack([
        xx*(xx*(xx*(xx*(-30.5963222836729*xx*y + 458.944834255093*y**3) - 1285.04553591426*y**5) + 917.889668510186*y**7) - 152.981611418364*y**9) + 2.78148384397026*y**11,
        x*(xx*(xx*(xx*(-130.463156574524*xx*y*z + 1565.55787889428*y**3*z) - 3287.671545678*y**5*z) + 1565.55787889428*y**7*z) - 130.463156574524*y**9*z),
        xx*(xx*(xx*(xx*(380.474049804873*xx*y + y*(-3170.61708170727*yy - 362.356237909402)) + y**3*(1775.54556575607*yy + 3381.99155382109)) + y**5*(3804.74049804873*yy - 5072.98733073163)) + y**7*(1449.42495163761 - 1479.62130479673*yy)) + y**9*(42.2748944227636*yy - 40.2618042121558),
        x*(xx*(xx*(xx*(124.746637761555*xx*y*z + y*(-748.479826569327*yy*z - 748.479826569327*z**3)) + 5239.35878598529*y**3*z**3) + y**5*(748.479826569327*yy*z - 5239.35878598529*z**3)) + y**7*(-124.746637761555*yy*z + 748.479826569327*z**3)),
        xx*(xx*(xx*(xx*(-1665.25883686909*xx*y + y*(4995.77651060727*yy + 2854.72943463273)) + y*(yy*(9991.55302121455*yy - 11418.9177385309) - 1201.99134089799)) + y**3*(yy*(-1427.36471731636*yy - 5709.45886926546) + 6009.95670448995)) + y**5*(yy*(8156.36981323637 - 4519.98827150182*yy) - 3605.97402269397)) + y**7*(yy*(237.894119552727*yy - 407.818490661818) + 171.713048699713),
        x*(xx*(xx*(xx*(-101.81331334922*xx*y*z + y*(135.751084465627*yy*z + 1086.00867572501*z**3)) + y*(yy*(475.128795629694*yy*z - 2534.02024335837*z**3) - 1520.41214601502*z**5)) + y**3*(yy*(135.751084465627*yy*z - 2534.02024335837*z**3) + 5068.04048671674*z**5)) + y**5*(yy*(-101.81331334922*yy*z + 1086.00867572501*z**3) - 1520.41214601502*z**5)),
        xx*(xx*(xx*(xx*(3798.86444744093*xx*y + y*(3798.86444744093*yy - 8683.11873700784)) + y*(6398.08749042683 - 10636.8204528346*y**4)) + y*(yy*(yy*(24312.7324636219 - 16715.0035687401*yy) - 6398.08749042683) - 1505.43235068867)) + y**3*(yy*(yy*(13892.9899792125 - 5318.4102264173*yy) - 11516.5574827683) + 3010.86470137733)) + y**5*(yy*(yy*(759.772889488186*yy - 1736.62374740157) + 1279.61749808537) - 301.086470137733),
        x*(xx*(xx*(xx*(71.1249931348854*xx*y*z + y*(142.249986269771*yy*z - 995.749903888396*z**3)) + y*(-995.749903888396*yy*z**3 + 2389.79976933215*z**5)) + y*(y**4*(-142.249986269771*yy*z + 995.749903888396*z**3) - 1137.99989015817*z**7)) + y**3*(yy*(yy*(-71.1249931348854*yy*z + 995.749903888396*z**3) - 2389.79976933215*z**5) + 1137.99989015817*z**7)),
        xx*(xx*(xx*(xx*(-4718.63793562039*xx*y + y*(12583.0344949877 - 17301.6724306081*yy)) + y*(yy*(33554.7586533006 - 22020.3103662285*yy) - 11920.7695215673)) + y*(yy*(yy*(25166.0689899754 - 9437.27587124079*yy) - 19867.9492026122) + 4674.81157708522)) + y*(yy*(yy*(1572.87931187346*y**4 - 3973.58984052244) + 3116.54105139015) - 623.308210278029)) + y**3*(yy*(yy*(yy*(1572.87931187346*yy - 4194.34483166257) + 3973.58984052244) - 1558.27052569507) + 207.76940342601),
        x*(xx*(xx*(xx*(-36.4407151441193*xx*y*z + y*(-145.762860576477*yy*z + 583.051442305908*z**3)) + y*(yy*(-218.644290864716*yy*z + 1749.15432691773*z**3) - 1749.15432691773*z**5)) + y*(yy*(yy*(-145.762860576477*yy*z + 1749.15432691773*z**3) - 3498.30865383545*z**5) + 1332.68901098493*z**7)) + y*(yy*(yy*(yy*(-36.4407151441193*yy*z + 583.051442305908*z**3) - 1749.15432691773*z**5) + 1332.68901098493*z**7) - 222.114835164156*z**9)),
        xx*(xx*(xx*(xx*(2236.7108242262*xx*y + y*(11183.554121131*yy - 6390.60235493199)) + y*(yy*(22367.108242262*yy - 25562.409419728) + 6726.94984729683)) + y*(yy*(yy*(22367.108242262*yy - 38343.6141295919) + 20180.8495418905) - 3165.62345755145)) + y*(yy*(yy*(yy*(11183.554121131*yy - 25562.409419728) + 20180.8495418905) - 6331.2469151029) + 633.12469151029)) + y*(yy*(yy*(yy*(yy*(2236.7108242262*yy - 6390.60235493199) + 6726.94984729683) - 3165.62345755145) + 633.12469151029) - 38.9615194775563),
        z*(zz*(zz*(zz*(zz*(4326.45824668537 - 1651.92042146169*zz) - 4098.74991791246) + 1687.72055443454) - 281.286759072424) + 12.9824658033426),
        x*(xx*(xx*(xx*(xx*(2236.7108242262*xx + 11183.554121131*yy - 6390.60235493199) + yy*(22367.108242262*yy - 25562.409419728) + 6726.94984729683) + yy*(yy*(22367.108242262*yy - 38343.6141295919) + 20180.8495418905) - 3165.62345755145) + yy*(yy*(yy*(11183.554121131*yy - 25562.409419728) + 20180.8495418905) - 6331.2469151029) + 633.12469151029) + yy*(yy*(yy*(yy*(2236.7108242262*yy - 6390.60235493199) + 6726.94984729683) - 3165.62345755145) + 633.12469151029) - 38.9615194775563),
        xx*z*(zz*(zz*(zz*(6725.91485231459 - 3923.45033051684*zz) - 3716.95294470017) + 728.814302882386) - 36.4407151441193) + z*(zz*(zz*(zz*(zz*(5324.68259141571 - 1961.72516525842*zz) - 5221.43389850738) + 2222.88362379128) - 382.627509013252) + 18.2203575720596),
        x*(xx*(xx*(xx*(xx*(-1572.87931187346*xx - 1572.87931187346*yy + 4194.34483166257) + 9437.27587124079*y**4 - 3973.58984052244) + yy*(yy*(22020.3103662285*yy - 25166.0689899754) + 3973.58984052244) + 1558.27052569507) + yy*(yy*(yy*(17301.6724306081*yy - 33554.7586533006) + 19867.9492026122) - 3116.54105139015) - 207.76940342601) + yy*(yy*(yy*(yy*(4718.63793562039*yy - 12583.0344949877) + 11920.7695215673) - 4674.81157708522) + 623.308210278029)),
        xx*(xx*z*(zz*(zz*(9189.3491130272 - 9189.3491130272*zz) - 2418.2497665861) + 142.249986269771) + z*(zz*(zz*(zz*(18378.6982260544 - 9189.3491130272*zz) - 11607.5988796133) + 2560.49975285588) - 142.249986269771)) + z*(zz*(zz*(zz*(zz*(3446.0059173852 - 1148.6686391284*zz) - 3748.28713820846) + 1771.01232905865) - 337.843717390706) + 17.7812482837214),
        x*(xx*(xx*(xx*(xx*(759.772889488186*xx - 5318.4102264173*yy - 1736.62374740157) + yy*(13892.9899792125 - 16715.0035687401*yy) + 1279.61749808537) + yy*(yy*(24312.7324636219 - 10636.8204528346*yy) - 11516.5574827683) - 301.086470137733) + yy*(yy*(3798.86444744093*y**4 - 6398.08749042683) + 3010.86470137733)) + y**4*(yy*(yy*(3798.86444744093*yy - 8683.11873700784) + 6398.08749042683) - 1505.43235068867)),
        xx*(xx*(xx*z*(zz*(6878.05494625843 - 14443.9153871427*zz) - 543.004337862507) + z*(zz*(zz*(31982.9555001017 - 21665.873080714*zz) - 11131.5889261814) + 814.506506793761)) + z*(zz*(zz*(zz*(20118.3107178059 - 8124.70240526777*zz) - 16167.9541598562) + 4479.78578736569) - 305.43994004766)) + z*(zz*(zz*(zz*(zz*(1569.0562846152 - 451.372355848209*zz) - 2015.90360431456) + 1147.09666373455) - 265.845873745186) + 16.9688855582034),
        x*(xx*(xx*(xx*(xx*(-237.894119552727*xx + 4519.98827150182*yy + 407.818490661818) + yy*(1427.36471731636*yy - 8156.36981323637) - 171.713048699713) + yy*(yy*(5709.45886926546 - 9991.55302121455*yy) + 3605.97402269397)) + y**4*(yy*(11418.9177385309 - 4995.77651060727*yy) - 6009.95670448995)) + y**6*(yy*(1665.25883686909*yy - 2854.72943463273) + 1201.99134089799)),
        xx*(xx*(xx*(xx*z*(1995.94620418487 - 13971.6234292941*zz) + z*(zz*(31935.139266958 - 27943.2468585882*zz) - 3991.89240836974)) + z*(zz*(zz*(37423.9913284664 - 17464.5292866176*zz) - 22454.3947970798) + 2494.93275523109)) + z*(zz*(zz*(zz*(10977.7041230168 - 3492.90585732353*zz) - 11975.6772251092) + 4989.86551046218) - 498.986551046218)) + z*(zz*(zz*(zz*(zz*(452.206561885635 - 109.15330804136*zz) - 717.293167128939) + 530.173210486607) - 171.526626922137) + 15.5933297201943),
        x*(xx*(xx*(xx*(xx*(42.2748944227636*xx - 1479.62130479673*yy - 40.2618042121558) + yy*(3804.74049804873*yy + 1449.42495163761)) + y**4*(1775.54556575607*yy - 5072.98733073163)) + y**6*(3381.99155382109 - 3170.61708170727*yy)) + y**8*(380.474049804873*yy - 362.356237909402)),
        xx*(xx*(xx*(xx*(-6679.71361661561*xx*z + z*(16699.284041539 - 16699.284041539*zz)) + z*(zz*(29223.7470726933 - 14611.8735363466*zz) - 14611.8735363466)) + z*(zz*(zz*(15655.5787889428 - 5218.52626298095*zz) - 15655.5787889428) + 5218.52626298095)) + z*(zz*(zz*(zz*(2609.26313149047 - 652.315782872618*zz) - 3913.89469723571) + 2609.26313149047) - 652.315782872618)) + z*(zz*(zz*(zz*(zz*(65.2315782872618 - 13.0463156574524*zz) - 130.463156574524) + 130.463156574524) - 65.2315782872618) + 13.0463156574524),
        x*(xx*(xx*(xx*(xx*(-2.78148384397026*xx + 152.981611418364*yy) - 917.889668510186*y**4) + 1285.04553591426*y**6) - 458.944834255093*y**8) + 30.5963222836729*y**10),
    ], dim=-1)


_ys = [y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11]


def spherical_harmonics(l, xyz, normalization='integral'):
    """
    spherical harmonics

    :param irreps: list of L's
    :param xyz: tensor of shape [..., 3]
    :param normalization: integral (the integral over the sphere gives 1), norm (the norm is 1), component (each component is ~1)
    :return: tensor of shape [..., m]
    """
    if isinstance(l, o3.Irreps):
        ls = [l for mul, (l, p) in l for _ in range(mul)]
    elif isinstance(l, int):
        ls = [l]
    else:
        ls = list(l)

    assert normalization in ['integral', 'component', 'norm']

    with torch.autograd.profiler.record_function(f'spherical_harmonics({ls}, {tuple(xyz.shape[:-1])})'):
        *size, _ = xyz.shape
        xyz = xyz.reshape(-1, 3)
        d = torch.norm(xyz, 2, dim=1)
        xyz = xyz[d > 0]
        xyz = xyz / d[d > 0, None]

        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        sh = torch.cat([_ys[l](x, y, z) for l in ls], dim=1)

        if len(d) > len(sh):
            out = sh.new_zeros(len(d), sh.shape[1])
            out[d == 0] = torch.cat([sh.new_ones(1) if l == 0 else sh.new_zeros(2 * l + 1) for l in ls])
            out[d > 0] = sh
            sh = out

        if normalization == 'integral':
            sh.div_(math.sqrt(4 * math.pi))
        if normalization == 'norm':
            sh.div_(torch.cat([math.sqrt(2 * l + 1) * sh.new_ones(2 * l + 1) for l in ls]))
        return sh.reshape(*size, sh.shape[1])
