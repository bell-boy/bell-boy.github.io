class BlogPost {
    constructor(title, description, date, link, id){
        this.title = title
        this.description = description
        this.date = date
        this.link = link
        this.id = id
    }   

    render() {
        let element = document.createElement("div")
        element.classList.add(..."card d-flex flex-column border rounded p-3".split(" "))

        let title_element = document.createElement("h2")
        title_element.classList.add(..."card-title m-0".split(" "))
        title_element.innerText = this.title

        let id_element = document.createElement("p")
        id_element.classList.add(..."small m-0".split(" "))
        id_element.innerText = this.id

        let description_element = document.createElement("p")
        description_element.innerText = this.description

        let link_element = document.createElement("a")
        link_element.classList.add("stretched-link")
        link_element.href = this.link

        element.appendChild(title_element)
        element.appendChild(id_element)
        element.appendChild(description_element)
        element.appendChild(link_element)
        return element
    }
}

let posts_container = document.querySelector("#posts-container")
let posts_array = []
fetch("./postmetadata.json").then((res) => {
    if(!res.ok) {
        throw new Error(res.status)
    }
    return res.json()
}).then((val) => {
    val.posts.forEach(element => {
        posts_array.push((new BlogPost(element.title, element.description, "", element.path, element.id)))
    })
    posts_array.reverse()
})

let first_post = 0
let ctx_len = 2

let loadNext = () => {
    first_post += ctx_len
    updateBttns()
    renderCtxWindow()
}

let loadPrev = () => {
    first_post -= ctx_len
    updateBttns()
    renderCtxWindow()
}

let updateBttns = () => {
    prevBtn = document.querySelector("#prev-btn")
    nextBtn = document.querySelector("#next-btn")

    if(first_post + ctx_len >= posts_array.length) {
        nextBtn.disabled = true;
    } else {
        nextBtn.disabled = false;
    }

    if(first_post - ctx_len < 0) {
        prevBtn.disabled = true;
    } else {
        prevBtn.disabled = false;
    }
}

let renderCtxWindow = () => {
    for (let child = posts_container.firstChild; posts_container.children.length > 1 ; child = posts_container.firstChild) {
        posts_container.removeChild(child)
    }
    posts_array.slice(first_post, Math.min(first_post + ctx_len, posts_array.length)).reverse().map((val) => val.render()).forEach((val) => {
        posts_container.insertBefore(val, posts_container.firstChild)
    })
}

window.onload = () => {
    updateBttns()
    renderCtxWindow()
}

