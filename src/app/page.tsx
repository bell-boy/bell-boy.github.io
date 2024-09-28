import ProjectCard from "@/components/ProjectCard";
import { ProjectDescription, projects } from "@/data/projects.d";

export default function Home() {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 auto-cols-auto gap-4 min-w-screen min-h-screen p-4">
      <header className="m-2">
        <h1 className=" text-[6rem] md:text-[10rem] font-semibold">femi bello</h1>
        <p className="text-lg">building things with machine learning</p>
        <p>Reach me at <a href="mailto:femi.bello@utexas.edu">femi.bello@utexas.edu</a></p>
      </header>
      <div className="col-span-2 grid lg:grid-cols-2 gap-2 auto-rows-min">
        <div className="col-span-2 flex flex-col">
          <h2 className="text-2xl flex-shrink-0">notable projects/writing</h2>
          <hr className="border-grey-300 my-4"/>
        </div>
        {projects.map((projectObject : ProjectDescription) => <ProjectCard {...projectObject} key={projectObject.name} />)}
      </div>
    </div>
  );
}

