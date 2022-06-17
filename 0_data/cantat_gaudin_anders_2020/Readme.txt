J/A+A/633/A99 Gaia DR2 open clusters in the Milky Way. II (Cantat-Gaudin+, 2020)
================================================================================
Clusters and mirages: cataloguing stellar aggregates in the Milky Way.
    Cantat-Gaudin T., Anders F.
    <Astron. Astrophys. 633, A99 (2020)>
    =2020A&A...633A..99C        (SIMBAD/NED BibCode)
================================================================================
ADC_Keywords: Clusters, open ; Milky Way
Keywords: open clusters and associations: general - Galaxy: stellar content

Abstract:
    Many of the open clusters listed in modern catalogues were first
    reported by visual astronomers as apparent over-densities of bright
    stars. As observational techniques and analysis methods improved, some
    of them have been shown to be chance alignments of stars and are not
    true clusters. Recent publications making use of Gaia DR2 data
    provided membership list for over a thousand clusters, but many nearby
    objects listed in the literature have so far evaded detection. We
    update the Gaia DR2 cluster census by performing membership
    determinations for known clusters that had been missed by previous
    studies, and for recently discovered clusters. We investigate a subset
    of non-detected clusters that according to their literature parameters
    should be easily visible in the Gaia . Confirming or disproving the
    existence of old, inner-disc, high-altitude clusters is especially
    important as their survival or disruption is linked to the dynamical
    processes that drive the evolution of the Milky Way. We employ the
    Gaia DR2 catalogue and a membership assignment procedure, as well as
    visual inspection of spatial, proper motion, and parallax
    distributions. We use membership lists provided by other authors when
    they are available. We derive membership lists for 150 objects,
    including 10 that were known prior to Gaia . We compile a final list
    of members for 1481 clusters. Among the objects that we are still
    unable to identify with Gaia data, we argue that many (mostly putative
    old, relatively nearby, high-altitude objects) are not true clusters.
    At present, the only confirmed cluster located further than 500pc
    away from the Galactic plane within the Solar circle is NGC 6791. It
    is likely that the objects discussed in this study only represent a
    fraction of the non-physical groupings erroneously listed in the
    catalogues as genuine open clusters, and that those lists need further
    cleaning.

Description:
    The catalogue ibcluses lists of cluster members for objects that were
    not included in Cantat-Gaudin et al. (2018, Cat. J/A+A/618/A93) either
    because they were not identified or because they had not yet been
    discovered. We bring the total number of clusters with available
    membership from Gaia DR2 to 1481.

    table1:
    Mean parameters of the identified clusters.

    members:
    All columns from the Gaia DR2 catalogue, except the membership
    probability which was computed with the unsupervised classification
    scheme UPMASK applied to the Gaia DR2 proper motions and parallaxes.

File Summary:
--------------------------------------------------------------------------------
 FileName      Lrecl  Records   Explanations
--------------------------------------------------------------------------------
ReadMe            80        .   This file
table1.dat       208     1481   Mean parameters of the identified clusters
members.dat      493   435833  *Members
--------------------------------------------------------------------------------
Note on members.dat: All columns from the Gaia DR2 catalogue, except the
 membership probability which was computed with the unsupervised classification
 scheme UPMASK applied to the Gaia DR2 proper motions and parallaxes.
--------------------------------------------------------------------------------

See also:
          I/345 : Gaia DR2 (Gaia Collaboration, 2018)
 J/A+A/618/A93  : Gaia DR2 open clusters in the Milky Way (Cantat-Gaudin+, 2018)
 J/A+A/623/A108 : Age of 269 GDR2 open clusters (Bossini+, 2019)

Byte-by-byte Description of file: table1.dat
--------------------------------------------------------------------------------
   Bytes Format Units    Label    Explanations
--------------------------------------------------------------------------------
   1- 17  A17   ---      Cluster  Cluster name (cluster)
  19- 25  F7.3  deg      RAdeg    [] Mean right ascension of members (ICRS)
                                   at Ep=2015.5 (ra)
  27- 33  F7.3  deg      DEdeg    Mean declination of members (ICRS)
                                   at Ep=2015.5  (dec)
  35- 41  F7.3  deg      GLON     Mean Galactic longitude of members (l)
  43- 49  F7.3  deg      GLAT     Mean Galactic latitude of members (b)
  51- 55  F5.3  deg      r50      Radius containing half the members (r50)
  57- 60  I4    ---      N        Number of members (probability over 0.5)
                                   (nbstars)
  62- 68  F7.3  mas/yr   pmRA     Mean proper motion along RA of members,
                                   pmRA*cosDE (pmra)
  70- 74  F5.3  mas/yr s_pmRA     Standard deviation of pmRA of members
                                   (sigpmra)
  76- 80  F5.3  mas/yr e_pmRA     Standard deviation of pmRA of members over
                                   square root of nbstars (uncertpmra)
  82- 88  F7.3  mas/yr   pmDE     Mean proper motion along DE of members (pmdec)
  90- 94  F5.3  mas/yr s_pmDE     Standard deviation of pmdec of members
                                   (sigpmdec)
  96-100  F5.3  mas/yr e_pmDE     Standard deviation of pmdec of members over
                                   square root of nbstars (uncertpmdec)
 102-107  F6.3  mas      Plx       Mean parallax of members (par)
 109-113  F5.3  mas    s_Plx      Standard deviation of parallax of members
                                   (sigpar)
 115-119  F5.3  mas    e_Plx      Standard deviation of parallax of members
                                   over square root of nbstars (uncertpar)
 121-127  F7.1  pc       dmode05  5th percentile distance confidence interval
                                   (d05)
 129-135  F7.1  pc       dmode16  16th percentile distance confidence interval
                                   (d16)
 137-143  F7.1  pc       dmode    Most likely distance (dmode)
 145-151  F7.1  pc       dmode84  84th percentile distance confidence interval
                                   (d84)
 153-159  F7.1  pc       dmode95  95th percentile distance confidence interval
                                   (d95)
 161-166  F6.1  pc       dmode+01 Distance if +0.1mas added to parallax
                                   (dmodePLUS01)
 168-174  F7.1  pc       dmode-01 ? Distance if -0.1mas added to parallax null
                                   if mean parallax under 0.1mas (dmodeMINUS01)
 176-183  F8.1  pc       X        X position in Galactic cartesian coordinates
                                   (X)
 185-192  F8.1  pc       Y        Y position in Galactic cartesian coordinates
                                   (Y)
 194-200  F7.1  pc       Z        Z position in Galactic cartesian coordinates
                                   (Z)
 202-208  F7.1  pc       Rgc      Distance from Galactic centre assuming the
                                   Sun is at 8340pc (Rgc)
--------------------------------------------------------------------------------

Byte-by-byte Description of file: members.dat
--------------------------------------------------------------------------------
   Bytes Format Units     Label       Explanations
--------------------------------------------------------------------------------
   1- 21 E21.18 deg       RAdeg       Barycentric right ascension (ICRS)
                                      at Ep=2015.5 (ra)
  23- 43 E21.18 deg       DEdeg       Barycentric declination (ICRS)
                                       at Ep=2015.5 (dec)
  45- 63  I19   ---       Source      Gaia DR2 source ID (source_id)
  65- 85 F21.17 deg       GLON        Galactic longitude (l)
  87-108 E22.19 deg       GLAT        Galactic latitude (b)
 110-131 E22.18 mas       Plx         Absolute stellar parallax (parallax)
 133-152 F20.18 mas     e_Plx         Standard error of parallax
                                       (parallax_error)
 154-175 E22.19 mas/yr    pmRA        Proper motion in right ascension direction
                                       (pmRA*cosDE) (pmra)
 177-196 F20.18 mas/yr  e_pmRA        Standard error of proper motion in right
                                       ascension direction (pmra_error)
 198-219 E22.19 mas/yr    pmDE        Proper motion in declination direction
                                       (pmdec)
 221-240 F20.18 mas/yr  e_pmDE        Standard error of proper motion in
                                       declination direction (pmdec_error)
 242-265 F24.19 km/s      RV          ? Spectroscopic radial velocity in the
                                       solar barycentric reference frame
                                       (radial_velocity)
 267-287 F21.17 km/s    e_RV          ? Radial velocity error
                                       (radial_velocity_error)

 289-302 E14.10 ---       RADEcor     Correlation between right ascension and
                                       declination (ra_dec_corr)
 304-317 E14.10 ---       RAPlxcor    Correlation between right ascension and
                                       parallax (ra_parallax_corr)
 319-332 E14.10 ---       RApmRAcor   Correlation between right ascension and
                                       proper motion in right ascension
                                       (ra_pmra_corr)
 334-347 E14.10 ---       RApmDEcor   Correlation between right ascension and
                                       proper motion in declination
                                       (ra_pmdec_corr)
 349-362 E14.10 ---       DEPlxcor    Correlation between declination and
                                       parallax (dec_parallax_corr)
 364-377 E14.10  ---      DEpmRAcor   Correlation between declination and
                                       proper motion in right ascension
                                       (dec_pmra_corr)
 379-392 E14.10 ---       DEpmDEcor   Correlation between declination and
                                       proper motion in declination
                                       (dec_pmdec_corr)
 394-407 E14.10  ---      PlxpmRAcor  Correlation between parallax and proper
                                       motion in right ascension
                                       (parallax_pmra_corr)
 409-422 E14.10 ---       PlxpmDEcor  Correlation between parallax and proper
                                       motion in declination
                                       (parallax_pmdec_corr)
 424-437 E14.10 ---       pmRApmDEcor Correlation between proper motion in
                                      right ascension and proper motion in
                                      declination (pmra_pmdec_corr)
 439-442  I4    ---     o_Gmag        Number of observations contributing to G
                                       photometry (phot_g_n_obs)
 444-453  F10.7 mag       Gmag        G-band mean magnitude (Vega)
                                       (phot_g_mean_mag)
 455-467 E13.10 mag       BP-RP       ? BP-RP colour (bp_rp)
 469-475  F7.5  ---       Proba       Membership probability (proba) (1)
 477-493  A17   ---       Cluster     Cluster name (cluster)


--------------------------------------------------------------------------------
Note (1): membership probability was computed with the unsupervised
  classification scheme UPMASK applied to the Gaia DR2 proper motions and
  parallaxes
--------------------------------------------------------------------------------

Acknowledgements:
     Tristan Cantat-Gaudin, tcantat(at)fqa.ub.edu

================================================================================
(End)                                        Patricia Vannier [CDS]  13-Nov-2019