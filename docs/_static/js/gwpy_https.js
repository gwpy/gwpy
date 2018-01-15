/* --------------------------------------------------------------------------*/
/* GWpy Docs force HTTPS                                                     */

// Credit: https://konklone.com/post/github-pages-now-supports-https-so-use-it

var enforce = "gwpy.github.io";
if ((enforce == window.location.host) && (window.location.protocol != "https:"))
  window.location.protocol = "https";
