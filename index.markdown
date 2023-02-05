---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Ditto in the House: Building Articulated Models of Indoor Scenes through Interactive Perception</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }
  
  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }
  
IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->

<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">
</head>

<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center><h1><strong>Ditto in the House: Building Articulated Models of Indoor Scenes through Interactive Perception</strong></h1></center>
<center><h2>
    <a href="https://chengchunhsu.github.io/">Cheng-Chun Hsu</a>&nbsp;&nbsp;&nbsp; 
    <a href="https://zhenyujiang.me/">Zhenyu Jiang</a>&nbsp;&nbsp;&nbsp;
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;&nbsp;
   </h2>
<center><h2>
    <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp; 		
</h2></center>
<center><h2>
        ICRA 2023&nbsp;&nbsp;&nbsp; 		
    </h2></center>
    <center><h2><a href="https://arxiv.org/abs/2302.01295">Paper</a> | <a href="">Code (Coming Soon)</a> </h2></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Virtualizing the physical world into virtual models has been a critical technique for robot navigation and planning in the real world. To foster manipulation with articulated objects in everyday life, this work explores building articulation models of indoor scenes through a robot's purposeful interactions in these scenes. Prior work on articulation reasoning primarily focuses on siloed objects of limited categories. To extend to room-scale environments, the robot has to efficiently and effectively explore a large-scale 3D space, locate articulated objects, and infer their articulations. We introduce an interactive perception approach to this task. Our approach, named Ditto in the House, discovers possible articulated objects through affordance prediction, interacts with these objects to produce articulated motions, and infers the articulation properties from the visual observations before and after each interaction. It tightly couples affordance prediction and articulation inference to improve both tasks. We demonstrate the effectiveness of our approach in both simulation and real-world scenes.
</p></td></tr></table>
</p>
</div>
</p>


<br><hr>
<h1 align="center">Problem Definition</h1>

<!-- <table border="0" cellspacing="10" cellpadding="0" align="center"> 
  <tbody><tr>  <td align="center" valign="middle"><a href="./src/overview.png"> <img src="./src/overview.png" style="width:100%;">  </a></td>
  </tr>
</tbody>
</table> -->

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
    <img src="./src/teaser.jpg" style="width:100%;">
    </td>
  </tr>
  </tbody>
</table>

  <table align=center width=800px>
                <tr>
                    <td>
  <p align="justify" width="20%">
  We explore the problem of building an articulation model of an indoor scene populated with articulated objects. An articulated object consists of multiple parts, and their connecting joints constrain the relative motion between each pair of parts. Scene-level articulated model informs the robot where the interactable regions are and gives more context about how to interact with the objects.
</p></td></tr></table>


<br><hr> <h1 align="center">Framework</h1> <!-- <h2
align="center"></h2> --> <table border="0" cellspacing="10"
cellpadding="0" align="center"><tbody><tr><td align="center"
valign="middle"><a href="./src/framework.jpg"> <img
src="./src/framework.jpg" style="width:100%;"> </a></td>
</tr> </tbody> </table>

<table width=800px><tr><td> <p align="justify" width="20%">Our approach consists of two stages --- affordance prediction and articulation inference. During affordance prediction, we pass the static scene point cloud into the affordance network and predict the scene-level affordance map. By applying point non-maximum suppression (NMS), we extract the interaction hotspots from the affordance map. Then, the robot interacts with the object based on those contact points. During articulation inference, we feed the point cloud observations before and after each interaction into the articulation model network to obtain articulation estimation. By aggregating the estimated articulation models, we build the articulation models of the entire scene.  </p></td></tr></table>
<br>


<br><hr>
<h1 align="center">Reconstruction Results</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">We show qualitative results on the iGibson scenes. Our model first opens the cabinet to a larger degree and reveals more previously occluded surfaces. With the new observation with more significant object state change, our refined model can predict more accurate part segmentation and joint parameters.
  </p>
</td></tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  <img src="./src/qual.jpg" width="100%">
  </td>
  </tr>

</tbody>
</table>


<br><hr>
<h1 align="center">Real World Experiment</h1>
<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p>We evaluate our method in a real-world household scene. We use the LiDAR and camera of an iPhone 12 Pro to recreate the scene in a 3D scan, rather than using a physical robot. We predict interaction hotspots and interact with the objects at these hotspots with our own hands. We then collect novel observations and run our approach to build the scene-level articulation model. The videos show that our approach can be applied to the real scenario without any modification and reconstruct an accurate articulation model of the scene. </p></td></tr></table>

<h1 align="center">Kitchen</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted controls width="100%">
        <source src="./video/real_kitchen.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>

<h1 align="center">Office</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted controls width="100%">
        <source src="./video/real_office.mp4"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table>


<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->


<hr />
<center><h1>Citation</h1></center>
<table align="center" width="800px">
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@misc{Hsu2023DittoITH,
  title={Ditto in the House: Building Articulation Models of Indoor Scenes through Interactive Perception},
  author={Cheng-Chun Hsu and Zhenyu Jiang and Yuke Zhu},
  publisher = {arXiv},
  year={2023}
}
</code></pre>
</left></td></tr></table>


<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-7GF0RHBSDK"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-7GF0RHBSDK');
</script>
<!-- </center></div></body></div> -->

