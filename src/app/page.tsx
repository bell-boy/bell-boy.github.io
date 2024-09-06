import ProjectCard from "@/components/ProjectCard";

export default function Home() {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 auto-cols-auto gap-4 min-w-screen min-h-screen p-4">
      <header className="m-2">
        <h1 className=" text-[6rem] md:text-[10rem] font-semibold">femi bello</h1>
        <p className="text-lg">building things with machine learning</p>
        <p>Reach me at <a href="mailto:femi.bello@utexas.edu">femi.bello@utexas.edu</a></p>
      </header>
      <div className="">
        <h2 className="text-2xl">notable projects</h2>
        <hr className="border-grey-300 my-4"/>
        <ProjectCard descriptionText="This is a test description" title="Test" date={new Date()}/>
      </div>
      <div className="min-w-[30em]">
        <h2 className="text-2xl">recent blogs</h2>
        <hr className="border-grey-300 my-4"/>
      </div>
    </div>
  );
}

