#!/bin/bash
# Copyright 2020 Google and DeepMind.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

REPO=$PWD
DIR=$REPO/download/
mkdir -p $DIR



# Helper function to download the UD-POS data.
# In order to ensure backwards compatibility with the XTREME evaluation,
# languages in XTREME use the UD version used in the original paper; for the new
# languages in XTREME-R, we use a more recent UD version.
function download_treebank {
    base_dir=$2
    out_dir=$3
    if [ $1 == "xtreme" ]; then
      url=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3105/ud-treebanks-v2.5.tgz
      langs=(af ar bg de el en es et eu fa fi fr he hi hu id it ja kk ko mr nl pt ru ta te th tl tr ur vi yo zh)
      ud_version="2.5"
    elif [ $1 == "xtreme-r" ]; then
      url=https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
      langs=(lt pl uk wo ro)
      ud_version="2.7"
    else
      echo "$1 is not an accepted argument for downloading the treebank. Accepted values: xtreme, xtreme-r"
      exit 0
    fi
    echo $1
    echo "$url"
    curl -s --remote-name-all "$url"

    tar -xzf $base_dir/ud-treebanks-v$ud_version.tgz

    for x in $base_dir/ud-treebanks-v$ud_version/*/*.conllu; do
        file="$(basename $x)"
        IFS='_' read -r -a array <<< "$file"
        lang=${array[0]}
        if [[ " ${langs[@]} " =~ " ${lang} " ]]; then
            lang_dir=$out_dir/$lang/
            mkdir -p $lang_dir
            y=$lang_dir/${file/conllu/conll}
            if [ ! -f "$y" ]; then
                echo "python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms"
                python $REPO/third_party/ud-conversion-tools/conllu_to_conll.py $x $y --lang $lang --replace_subtokens_with_fused_forms --print_fused_forms
            else
                echo "${y} exists"
            fi
        fi
    done
}

# Download UD-POS dataset.
function download_udpos {
    base_dir=$DIR/udpos-tmp
    out_dir=$base_dir/conll/
    mkdir -p $out_dir
    cd $base_dir

    download_treebank xtreme $base_dir $out_dir
    download_treebank xtreme-r $base_dir $out_dir

    python $REPO/utils_preprocess.py --data_dir $out_dir/ --output_dir $DIR/udpos/ --task  udpos
    rm -rf $out_dir ud-treebanks-v2.tgz $DIR/udpos-tmp
    echo "Successfully downloaded data at $DIR/udpos" >> $DIR/download.log
}

function download_panx {
    echo "Download panx NER dataset"
    if [ -f $DIR/AmazonPhotos.zip ]; then
        base_dir=$DIR/panx_dataset/
        unzip -qq -j $DIR/AmazonPhotos.zip -d $base_dir
        cd $base_dir
        langs=(ar he vi id jv ms tl eu ml ta te af nl en de el bn hi mr ur fa fr it pt es bg ru ja ka ko th sw yo my zh kk tr et fi hu qu pl uk az lt pa gu ro)
        for lg in ${langs[@]}; do
            tar xzf $base_dir/${lg}.tar.gz
            for f in dev test train; do mv $base_dir/$f $base_dir/${lg}-${f}; done
        done
        cd ..
        python $REPO/utils_preprocess.py \
            --data_dir $base_dir \
            --output_dir $DIR/panx \
            --task panx
        rm -rf $base_dir
        echo "Successfully downloaded data at $DIR/panx" >> $DIR/download.log
    else
        echo "Please download the AmazonPhotos.zip file on Amazon Cloud Drive manually and save it to $DIR/AmazonPhotos.zip"
        echo "https://www.amazon.com/clouddrive/share/d3KGCRCIYwhKJF0H3eWA26hjg2ZCRhjpEQtDL70FSBN"
    fi
}



download_udpos
download_panx
