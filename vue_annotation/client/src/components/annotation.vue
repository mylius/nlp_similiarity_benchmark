<template>
<div class="sentences">
    <div>
    <h3 >Reference sentence:</h3>
    <p>{{sentences[ids[0]].sent}}</p>
    </div>
    <div class="comp">
    <div>
    <h3>Sentence 1:</h3>
    <p>{{sentences[ids[1]].sent}}</p>
    </div>
    <div>
    <h3>Sentence 2:</h3>
    <p>{{sentences[ids[2]].sent}}&nbsp;</p>
    </div>
    </div>
    &nbsp;
    &nbsp;
    Which sentence is more similar to the reference sentence?
    &nbsp;  
    <div>
    <button v-on:click="select1">1</button>
    <button v-on:click="select2">2</button>
    <button v-on:click="skip">skip</button>
    </div>
</div>
</template>

<script>
import DataService from '@/services/dataservice';
export default {
    name: 'annotation',
    data(){
        return {
            sentences: [],
            error: '',
            text: '',
            ids: [...Array(3).keys()],
        }
    },
    async created() {
        try{
            this.sentences = await DataService.getSentences();
            for (var sent in this.sentences){
                this.sentences[sent].annotations = {};
            }
            this.setIds();
        } catch(err) {
            this.error = err.message;
        }
        
    },
    methods: {
        async updateSentence () {
            await DataService.updateSentence({
                id:  this.sentences[this.ids[0]].id,
                sent: this.sentences[this.ids[0]].sent,
                annotations: this.sentences[this.ids[0]].annotations
            })
        },
        randomChoiceNoReplace(InArray, len) {
            var bucket = [];
            var OutArray = new Array(len).fill(null);
            var randomIndex = 0;

            for (let i in OutArray) {
            bucket.push(InArray[i]);
            }
            for (let entry in OutArray) {
            randomIndex = Math.floor(Math.random() * bucket.length);
            OutArray[entry] = bucket.splice(randomIndex, 1)[0];
            }
            return OutArray
        },
        setIds() {
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
        },
        select1() {
            this.annotate(1,2); 
            this.updateSentence();
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
            },
        select2() {
            this.annotate(2,1);
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
        },
        skip() {
            this.ids = this.randomChoiceNoReplace([...Array(this.sentences.length).keys()], 3);
        },
        annotate(a,b) {
            if (this.sentences[this.ids[0]].annotations["("+this.ids[a]+","+this.ids[b]+")"] == null){
               this.sentences[this.ids[0]].annotations["("+this.ids[a]+","+this.ids[b]+")"] = 1
            } else{
                this.sentences[this.ids[0]].annotations["("+this.ids[a]+","+this.ids[b]+")"] += 1;
            }
        }
    }
}

</script>
<style scoped>
.sentences {
    display: grid;
    grid-template-columns: repeat(1,1fr);
    grid-auto-rows: minmax(50px, auto);
}
.comp {
    display: grid;
    grid-template-columns: repeat(2,1fr);
    grid-auto-rows: minmax(50px, auto);
}
</style>