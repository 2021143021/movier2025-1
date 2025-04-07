let MovieObject = {
    init: function(){
        
        },

        getall: function(){
           
            $.ajax({

                type: "GET",
                url: "http://localhost:8000/all",
                }).done(function(response){

                    console.log(response)
                    movielist = response.result

                    topdiv = document.getElementById("alldiv")

                    //topdiv = document.createElement("div")
                    //topdiv.style = "column-count:5"
                    //document.body.appendChild(topdiv)


                    movielist.forEach(movie => {
                        cmovie = document.createElement("div")
                        cmovie.className = "card"
    
                        mimg = document.createElement("img")
                        mimg.className = "card-img-top"
                        mimg.src = movie.poster_path
                        mimg.onclick = function(){
                            // location.href = movie.url // 현재창에서 열기
                            window.open(movie.url) // 새창에서 열리기
                        }
                        mimg.onmouseover = function(){
                            mimg.style.cursor = "pointer"
                        }
                        cmovie.appendChild(mimg)
                        topdiv.appendChild(cmovie)
    
                    });

                    //첫번째 영화 이미지
                 
                   

                }).fail(function(error){

                    console.log(error)
                    });
                },

            getgenres: function(){
                
                genre = document.getElementById("sgenre").value

                $.ajax({
    
                    type: "GET",
                    url: "http://localhost:8000/genresbest/" + genre, 
                    }).done(function(response){
    
                        console.log(response)
                        movielist = response.result
    
                        topdiv = document.getElementById("genrediv")
                     //   topdiv.innerHTML = "" //기존 영화내용을 지우는 첫번째 방법
                        while(topdiv.firstChild){
                            topdiv.removeChild(topdiv.firstChild)
                        }
                
    
                        movielist.forEach(movie => {
                            cmovie = document.createElement("div")
                            cmovie.className = "card"
        
                            mimg = document.createElement("img")
                            mimg.className = "card-img-top"
                            mimg.src = movie.poster_path
                            mimg.onclick = function(){
                                // location.href = movie.url // 현재창에서 열기
                                window.open(movie.url) // 새창에서 열리기
                            }
                            mimg.onmouseover = function(){
                                mimg.style.cursor = "pointer"
                            }
                            cmovie.appendChild(mimg)
                            topdiv.appendChild(cmovie)
        
                        });
    
                        //첫번째 영화 이미지
                     
                       
    
                    }).fail(function(error){
    
                        console.log(error)
                        });
                    }
                }
MovieObject.init();
MovieObject.getall();
                