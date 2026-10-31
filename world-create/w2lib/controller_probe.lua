-- Synthetic road probe; the game controller/tuning are loaded read-only.
-- Each row: layout x, layout y, height, width, grip, target speed.
local control=require "game.core.kart.controller"
local tuning=require "game.core.kart.tuning"
local util=require "game.core.math"
local results={}
for _,route in ipairs(routes) do
    local pts=route.pts
    local n=#pts
    local index=1
    local traversed=0
    local min_margin=math.huge
    local max_error=0
    local max_steer=0
    local airborne=0
    local function query(x,z)
        local best=math.huge
        local bi,bf,bl
        for shift=-6,18 do
            local i=(index-1+shift)%n+1
            local a,b=pts[i],pts[i%n+1]
            local dx,dz=b[1]-a[1],b[2]-a[2]
            local len2=dx*dx+dz*dz
            local f=util.clamp(((x-a[1])*dx+(z-a[2])*dz)/len2,0,1)
            local ex,ez=x-a[1]-f*dx,z-a[2]-f*dz
            local d=ex*ex+ez*ez
            if d<best then best,bi,bf,bl=d,i,f,math.sqrt(len2) end
        end
        return bi,bf,bl,math.sqrt(best)
    end
    local a,b=pts[1],pts[2]
    local k=control.new({x=a[1],y=a[3]+tuning.ride_height,z=a[2],yaw=math.atan(b[1]-a[1],b[2]-a[2])})
    local ground_index,ground_fraction=1,0
    local env={surface=function() return "road" end,ground=function(x,y,z)
        local i,f=query(x,z)
        local aa,bb=pts[i],pts[i%n+1]
        return aa[3]+(bb[3]-aa[3])*f,0
    end}
    local elapsed=0
    for tick=1,24000 do
        local i,f,len,error=query(k.x,k.z)
        local advance=(i-index+n)%n
        if advance<n/2 then traversed=traversed+advance end
        index=i
        local p=pts[i]
        local width=p[4]+(pts[i%n+1][4]-p[4])*f
        min_margin=math.min(min_margin,width/2-1.1-.3-error)
        max_error=math.max(max_error,error)
        local look=6+0.24*k.speed
        local j,rest=i,look+f*len
        while rest>0 do
            local aa,bb=pts[j],pts[j%n+1]
            local ll=math.sqrt((bb[1]-aa[1])^2+(bb[2]-aa[2])^2)
            if rest<=ll then
                local t=rest/ll
                ground_index,ground_fraction=j,t
                break
            end
            rest=rest-ll;j=j%n+1
        end
        local aa,bb=pts[ground_index],pts[ground_index%n+1]
        local target_x=aa[1]+(bb[1]-aa[1])*ground_fraction
        local target_z=aa[2]+(bb[2]-aa[2])*ground_fraction
        local angle=util.angle(math.atan(target_x-k.x,target_z-k.z)-k.yaw)
        local steer=util.clamp(angle*2.6,-1,1)
        max_steer=math.max(max_steer,math.abs(steer))
        local target=p[6]
        local th,br=1,0
        if k.speed>target+.3 then th,br=0,util.clamp((k.speed-target)*.4,0,1) end
        k.grip_scale=p[5]
        control.step(k,{throttle=th,brake=br,steer=steer,drift=false,jump=false},tuning,
            {speed_multiplier=1,grip_multiplier=1},env,1/60,function() end)
        if not k.grounded then airborne=airborne+1 end
        elapsed=tick/60
        if traversed>=n then break end
    end
    results[#results+1]=string.format('{"route":"%s","completed":%s,"time_s":%.3f,"min_margin_m":%.4f,"max_centre_error_m":%.4f,"max_steer":%.4f,"airborne_frames":%d}',
        route.id,tostring(traversed>=n),elapsed,min_margin,max_error,max_steer,airborne)
end
return "["..table.concat(results,",").."]"
