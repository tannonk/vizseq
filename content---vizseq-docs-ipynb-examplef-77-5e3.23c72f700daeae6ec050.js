(window.webpackJsonp=window.webpackJsonp||[]).push([[6],{152:function(e,t,a){"use strict";a.r(t),a.d(t,"frontMatter",(function(){return l})),a.d(t,"rightToc",(function(){return o})),a.d(t,"default",(function(){return p}));a(59),a(32),a(23),a(24),a(60),a(0);var n=a(164);function r(){return(r=Object.assign||function(e){for(var t=1;t<arguments.length;t++){var a=arguments[t];for(var n in a)Object.prototype.hasOwnProperty.call(a,n)&&(e[n]=a[n])}return e}).apply(this,arguments)}var l={id:"ipynb_example",title:"Jupyter Notebook Example",sidebar_label:"Jupyter Notebook Example"},o=[{value:"Example Data",id:"example-data",children:[]},{value:"Data Sources",id:"data-sources",children:[]},{value:"Viewing Examples and Statistics",id:"viewing-examples-and-statistics",children:[]},{value:"Google Translate Integration",id:"google-translate-integration",children:[]},{value:"Fairseq Integration",id:"fairseq-integration",children:[]},{value:"More Examples",id:"more-examples",children:[]}],i={rightToc:o},s="wrapper";function p(e){var t=e.components,a=function(e,t){if(null==e)return{};var a,n,r={},l=Object.keys(e);for(n=0;n<l.length;n++)a=l[n],t.indexOf(a)>=0||(r[a]=e[a]);return r}(e,["components"]);return Object(n.b)(s,r({},i,a,{components:t,mdxType:"MDXLayout"}),Object(n.b)("h2",{id:"example-data"},"Example Data"),Object(n.b)("p",null,"To get the data for the following examples:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-bash"}),"$ git clone https://github.com/facebookresearch/vizseq\n$ cd vizseq\n$ bash get_example_data.sh\n")),Object(n.b)("p",null,"The data will be available in ",Object(n.b)("inlineCode",{parentName:"p"},"examples/data"),"."),Object(n.b)("h2",{id:"data-sources"},"Data Sources"),Object(n.b)("p",null,"VizSeq accepts data from various types of sources: plain text file paths, ZIP file paths and Python dictionaries.\n(See also the ",Object(n.b)("a",r({parentName:"p"},{href:"data_sources"}),"data sources")," page for more details.)"),Object(n.b)("p",null,"Here is an example for plain text file paths as inputs:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"from glob import glob\nroot = 'examples/data/translation_wmt14_en_de_test'\nsrc = glob(f'{root}/src_*.txt')\nref = glob(f'{root}/ref_*.txt')\nhypo = glob(f'{root}/pred_*.txt')\n")),Object(n.b)("p",null,"An example for Python dictionaries as inputs:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"from typing import List, Dict\nimport os.path as op\nfrom glob import glob\n\ndef reader(paths: List[str]) -> Dict[str, List[str]]:\n    data = {}\n    for path in paths:\n        name = str(op.splitext(op.basename(path))[0]).split('_', 1)[1]\n        with open(path) as f:\n            data[name] = [l.strip() for l in f]\n    return data\n\nroot = 'examples/data/translation_wmt14_en_de_test'\nsrc = reader(glob(f'{root}/src_*.txt'))\nref = reader(glob(f'{root}/ref_*.txt'))\nhypo = reader(glob(f'{root}/pred_*.txt'))\n")),Object(n.b)("h2",{id:"viewing-examples-and-statistics"},"Viewing Examples and Statistics"),Object(n.b)("p",null,"Please see the ",Object(n.b)("a",r({parentName:"p"},{href:"ipynb_api"}),"Jupyter Notebook API doc")," for full references."),Object(n.b)("p",null,"First, load the ",Object(n.b)("inlineCode",{parentName:"p"},"vizseq")," package:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"import vizseq\n")),Object(n.b)("p",null,"To view dataset statistics:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"vizseq.view_stats(src, ref)\n")),Object(n.b)("p",null,"To view source-side n-grams:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"vizseq.view_n_grams(src)\n")),Object(n.b)("p",null,"To view corpus-level scores (BLEU and METEOR):"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"vizseq.view_scores(ref, hypo, ['bleu', 'meteor'])\n")),Object(n.b)("p",null,"To check the IDs of available scorers in VizSeq:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"vizseq.available_scorers()\n")),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{}),"Available scorers: bert_score, bleu, bp, chrf, cider, gleu, laser, meteor, nist, ribes, rouge_1, rouge_2, rouge_l, ter, wer, wer_del, wer_ins, wer_sub\n")),Object(n.b)("p",null,"We can view examples in pages with sorting:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"import vizseq.VizSeqSortingType\nvizseq.view_examples(src, ref, hypo, ['bleu'], page_sz=10, page_no=1, sorting=VizSeqSortingType.src_len)\n")),Object(n.b)("h2",{id:"google-translate-integration"},"Google Translate Integration"),Object(n.b)("p",null,"VizSeq integrates Google Translate using Google Cloud API, to use which you need a Google Cloud credential and let VizSeq know the credential JSON file path:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"vizseq.set_google_credential_path('path to google credential json file')\n")),Object(n.b)("p",null,"Then in example viewing API, simply switch the ",Object(n.b)("inlineCode",{parentName:"p"},"need_g_translate")," argument on:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"vizseq.view_examples(src, ref, hypo, ['bleu'], need_g_translate=True)\n")),Object(n.b)("h2",{id:"fairseq-integration"},"Fairseq Integration"),Object(n.b)("p",null,Object(n.b)("a",r({parentName:"p"},{href:"https://github.com/pytorch/fairseq"}),"Fairseq")," is a popular sequence modeling toolkit developed by Facebook AI Research.\nVizSeq can directly import and analyze model predictions generated by ",Object(n.b)("a",r({parentName:"p"},{href:"https://github.com/pytorch/fairseq/blob/master/generate.py"}),Object(n.b)("inlineCode",{parentName:"a"},"fairseq-generate"))," and ",Object(n.b)("a",r({parentName:"p"},{href:"https://github.com/pytorch/fairseq/blob/master/interactive.py"}),Object(n.b)("inlineCode",{parentName:"a"},"fairseq-interactive")),". The\nAPIs are almost the same:"),Object(n.b)("pre",null,Object(n.b)("code",r({parentName:"pre"},{className:"language-python"}),"from vizseq.ipynb import fairseq_viz as vizseq_fs\n\nlog_path = 'examples/data/wmt14_fr_en_test.fairseq_generate.log'\n\nvizseq_fs.view_stats(log_path)\nvizseq_fs.view_examples(log_path, ['bleu', 'meteor'], need_g_translate=True)\nvizseq_fs.view_scores(log_path, ['bleu', 'meteor'])\nvizseq_fs.view_n_grams(log_path)\n")),Object(n.b)("h2",{id:"more-examples"},"More Examples"),Object(n.b)("ul",null,Object(n.b)("li",{parentName:"ul"},Object(n.b)("a",r({parentName:"li"},{href:"examples/multimodal_machine_translation.ipynb"}),"Multimodal Machine Translation")),Object(n.b)("li",{parentName:"ul"},Object(n.b)("a",r({parentName:"li"},{href:"examples/multilingual_machine_translation.ipynb"}),"Multilingual Machine Translation")),Object(n.b)("li",{parentName:"ul"},Object(n.b)("a",r({parentName:"li"},{href:"examples/speech_translation.ipynb"}),"Speech Translation"))))}p.isMDXComponent=!0},164:function(e,t,a){"use strict";a.d(t,"a",(function(){return i})),a.d(t,"b",(function(){return b}));var n=a(0),r=a.n(n),l=r.a.createContext({}),o=function(e){var t=r.a.useContext(l),a=t;return e&&(a="function"==typeof e?e(t):Object.assign({},t,e)),a},i=function(e){var t=o(e.components);return r.a.createElement(l.Provider,{value:t},e.children)};var s="mdxType",p={inlineCode:"code",wrapper:function(e){var t=e.children;return r.a.createElement(r.a.Fragment,{},t)}},c=Object(n.forwardRef)((function(e,t){var a=e.components,n=e.mdxType,l=e.originalType,i=e.parentName,s=function(e,t){var a={};for(var n in e)Object.prototype.hasOwnProperty.call(e,n)&&-1===t.indexOf(n)&&(a[n]=e[n]);return a}(e,["components","mdxType","originalType","parentName"]),c=o(a),b=n,u=c[i+"."+b]||c[b]||p[b]||l;return a?r.a.createElement(u,Object.assign({},{ref:t},s,{components:a})):r.a.createElement(u,Object.assign({},{ref:t},s))}));function b(e,t){var a=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var l=a.length,o=new Array(l);o[0]=c;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i[s]="string"==typeof e?e:n,o[1]=i;for(var b=2;b<l;b++)o[b]=a[b];return r.a.createElement.apply(null,o)}return r.a.createElement.apply(null,a)}c.displayName="MDXCreateElement"}}]);