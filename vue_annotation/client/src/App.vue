<template>
  <div id="app">
    <annotation/>
  </div>
</template>

<script>
import annotation from './components/annotation.vue'

export default {
  name: 'Annotation',
  components: {
    annotation
  }
}
</script>

<style>
#app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
}
</style>
